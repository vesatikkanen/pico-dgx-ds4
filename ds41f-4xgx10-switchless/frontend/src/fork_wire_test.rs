mod fork_wire {
    use super::*;
    use crate::encoding::{ASSISTANT_SP, BOS, THINK_END, USER_SP};
    use crate::protocol::{read_frame, Done, Frame, HelloEngine, VERSION};
    use crate::tokenizer::bytes_to_unicode;
    use std::path::PathBuf;
    use tokio::io::AsyncWriteExt;
    use tokio::net::{UnixListener, UnixStream};
    use tokio::task::JoinHandle;
    use tokio::time::timeout;

    const DEADLINE: Duration = Duration::from_secs(5);

    struct SocketPath(PathBuf);

    impl Drop for SocketPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn tokenizer() -> Arc<Tokenizer> {
        let vocab: serde_json::Map<String, Value> = bytes_to_unicode()
            .iter().enumerate().map(|(id, c)| (c.to_string(), json!(id))).collect();
        let added: Vec<_> = [BOS, USER_SP, ASSISTANT_SP, THINK_END]
            .iter().enumerate().map(|(id, text)| {
                json!({"id": 256 + id, "content": text, "special": true, "normalized": false})
            }).collect();
        let tokenizer = json!({
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": added,
        });
        Arc::new(Tokenizer::from_json(&tokenizer.to_string()).unwrap())
    }

    async fn receive(peer: &mut UnixStream) -> Frame {
        let (ty, payload) = timeout(DEADLINE, read_frame(peer))
            .await.expect("frontend did not send a frame").unwrap().unwrap();
        Frame::decode(ty, &payload, |_| 0).unwrap()
    }

    async fn submit(peer: &mut UnixStream) -> Submit {
        match receive(peer).await {
            Frame::Submit(s) => s,
            frame => panic!("expected SUBMIT, got {frame:?}"),
        }
    }

    async fn connect(listener: &UnixListener, engine: &EngineClient) -> UnixStream {
        let (mut peer, _) = timeout(DEADLINE, listener.accept()).await.unwrap().unwrap();
        assert_eq!(receive(&mut peer).await, Frame::Hello { version: VERSION });
        peer.write_all(&Frame::HelloEngine(HelloEngine {
            version: VERSION,
            max_context: 4096,
            vocab_size: 260,
            eos_id: 0,
            max_sessions: 8,
            name: "synthetic".into(),
        }).encode()).await.unwrap();
        assert!(engine.wait_connected(DEADLINE).await.is_some());
        peer
    }

    async fn done(peer: &mut UnixStream, s: &Submit, reason: DoneReason) {
        peer.write_all(&Frame::Done(Done {
            req_id: s.req_id,
            reason,
            prompt_tokens: s.prompt.len() as u32,
            cached_tokens: 0,
            completion_tokens: 0,
            queue_us: 0,
            prefill_us: 0,
            decode_us: 0,
            error: if reason == DoneReason::Error { "synthetic failure".into() } else { String::new() },
        }).encode()).await.unwrap();
    }

    fn request(app: &Arc<App>, message: String) -> JoinHandle<Response> {
        let body = json!({"messages": [{"role": "user", "content": message}], "max_tokens": 1});
        tokio::spawn(chat(State(app.clone()), HeaderMap::new(), Bytes::from(body.to_string())))
    }

    async fn exercise(terminal: Option<DoneReason>) {
        let path = SocketPath(std::env::temp_dir().join(format!(
            "pico-fork-{}-{terminal:?}.sock", std::process::id()
        )));
        let listener = UnixListener::bind(&path.0).unwrap();
        let engine = EngineClient::start(path.0.clone());
        let mut peer = connect(&listener, &engine).await;
        let mut app = App::new(cfg(), tokenizer(), engine);
        *Arc::get_mut(&mut app).unwrap().sessions.get_mut().unwrap() = SessionTable::new(1, 4096);

        let prefix = "shared synthetic prefix ".repeat(8);
        let request_parent = request(&app, format!("{prefix}{}", "a".repeat(64)));
        let parent = submit(&mut peer).await;
        assert_eq!(parent.base_session_id, 0);
        done(&mut peer, &parent, DoneReason::Length).await;
        assert_eq!(timeout(DEADLINE, request_parent).await.unwrap().unwrap().status(), StatusCode::OK);

        let request_fork = request(&app, format!("{prefix}{}", "b".repeat(64)));
        let fork = submit(&mut peer).await;
        assert_eq!(fork.base_session_id, parent.session_id);
        assert_ne!(fork.session_id, parent.session_id);
        assert!(fork.prefix_hint > 0);
        request_fork.abort();
        assert!(timeout(DEADLINE, request_fork).await.unwrap().unwrap_err().is_cancelled());
        assert_eq!(receive(&mut peer).await, Frame::Cancel { req_id: fork.req_id });

        // This SUBMIT is a wire barrier. A parent RELEASE queued by the fork plan, by dropping the
        // HTTP request or by planning another request would be read before it.
        let request_other = request(&app, "unrelated request".into());
        let other = submit(&mut peer).await;
        assert_eq!(other.base_session_id, 0);

        if let Some(reason) = terminal {
            done(&mut peer, &fork, reason).await;
            if reason == DoneReason::Error {
                assert_eq!(receive(&mut peer).await, Frame::Release { session_id: fork.session_id });
            }
            assert_eq!(receive(&mut peer).await, Frame::Release { session_id: parent.session_id });
            done(&mut peer, &other, DoneReason::Length).await;
            assert_eq!(timeout(DEADLINE, request_other).await.unwrap().unwrap().status(), StatusCode::OK);
        } else {
            drop(peer);
            assert_eq!(
                timeout(DEADLINE, request_other).await.unwrap().unwrap().status(),
                StatusCode::INTERNAL_SERVER_ERROR,
            );
            // A fresh HELLO means the failed connection's entire pending map was drained. Both
            // failed destinations are gone even though the fork's HTTP receiver had been dropped.
            let mut peer = connect(&listener, &app.engine).await;
            assert!(app.sessions.lock().unwrap().len() <= 1);
            let retry = request(&app, format!("{prefix}{}", "b".repeat(64)));
            let replanned = submit(&mut peer).await;
            assert_ne!(replanned.session_id, fork.session_id);
            // The old base may itself have been evicted while clearing the failed requests.
            assert!(replanned.base_session_id == 0 || replanned.base_session_id == parent.session_id);
            done(&mut peer, &replanned, DoneReason::Length).await;
            assert_eq!(timeout(DEADLINE, retry).await.unwrap().unwrap().status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn queued_fork_keeps_base_until_cancelled_done() {
        exercise(Some(DoneReason::Cancelled)).await;
    }

    #[tokio::test]
    async fn queued_fork_keeps_base_until_error_done() {
        exercise(Some(DoneReason::Error)).await;
    }

    #[tokio::test]
    async fn disconnect_cleans_up_fork_without_http_receiver() {
        exercise(None).await;
    }

    async fn reasoning_loop_response(stream: bool) {
        use crate::protocol::Tokens;

        let path = SocketPath(std::env::temp_dir().join(format!(
            "pico-reasoning-loop-{}-{stream}.sock", std::process::id()
        )));
        let listener = UnixListener::bind(&path.0).unwrap();
        let engine = EngineClient::start(path.0.clone());
        let mut peer = connect(&listener, &engine).await;
        let mut config = cfg();
        config.loop_guard = Some(LoopGuard::default());
        let app = App::new(config, tokenizer(), engine);
        let body = json!({
            "messages": [{"role": "user", "content": "Read the documentation"}],
            "thinking": {"type": "enabled"}, "stream": stream, "max_tokens": 2048,
            "tools": [{"type": "function", "function": {"name": "read_doc",
                "description": "Read a document", "parameters": {"type": "object", "properties": {}}}}]
        });
        let request = tokio::spawn({
            let app = app.clone();
            async move {
                let response = chat(State(app), HeaderMap::new(), Bytes::from(body.to_string())).await;
                let status = response.status();
                let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
                (status, bytes)
            }
        });
        let submitted = submit(&mut peer).await;
        let lines = ["Let me read the docs.", "OK.", "Let me write the calls.",
            "Making calls.", "Let me go.", "Writing now."];
        let text = (0..24).map(|i| format!("{}\n\n", lines[i % lines.len()])).collect::<String>();
        let ids = app.tok.encode_parallel(&text, 1);
        for chunk in ids.chunks(if stream { 37 } else { ids.len() }) {
            peer.write_all(&Frame::Tokens(Tokens {
                req_id: submitted.req_id, ids: chunk.to_vec(), logprobs: None, top: None,
            }).encode()).await.unwrap();
        }
        assert_eq!(receive(&mut peer).await, Frame::Cancel { req_id: submitted.req_id });
        done(&mut peer, &submitted, DoneReason::Cancelled).await;
        assert_eq!(receive(&mut peer).await, Frame::Release { session_id: submitted.session_id });

        let (status, bytes) = timeout(DEADLINE, request).await.unwrap().unwrap();
        assert_eq!(status, if stream { StatusCode::OK } else { StatusCode::INTERNAL_SERVER_ERROR });
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("repetition loop detected"), "{body}");
        assert!(!body.contains("\"finish_reason\":\"stop\""), "{body}");
        if stream {
            assert!(body.contains("[DONE]"));
        }
        assert_eq!(app.sessions.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn reasoning_loop_returns_error_and_releases_session() {
        reasoning_loop_response(false).await;
        reasoning_loop_response(true).await;
    }

    #[tokio::test]
    async fn reasoning_detector_respects_scope_and_disable_flag() {
        use crate::protocol::Tokens;

        for (tools, thinking, enabled) in [(false, true, true), (true, false, true), (true, true, false)] {
            let path = SocketPath(std::env::temp_dir().join(format!(
                "pico-reasoning-scope-{}-{tools}-{thinking}-{enabled}.sock", std::process::id()
            )));
            let listener = UnixListener::bind(&path.0).unwrap();
            let engine = EngineClient::start(path.0.clone());
            let mut peer = connect(&listener, &engine).await;
            let mut config = cfg();
            config.loop_guard = enabled.then(LoopGuard::default);
            let app = App::new(config, tokenizer(), engine);
            let mut body = json!({
                "messages": [{"role": "user", "content": "Read the documentation"}],
                "thinking": {"type": if thinking { "enabled" } else { "disabled" }},
                "max_tokens": 2048
            });
            if tools {
                body["tools"] = json!([{"type": "function", "function": {"name": "read_doc",
                    "parameters": {"type": "object", "properties": {}}}}]);
            }
            let request = tokio::spawn({
                let app = app.clone();
                async move { chat(State(app), HeaderMap::new(), Bytes::from(body.to_string())).await }
            });
            let submitted = submit(&mut peer).await;
            let text = "Making calls.\nLet me write.\nOK.\nLet me just do it.\nReading docs.\n\
                Let me write the calls.\nOK.\nLet me write.\nI'll read the two docs.\n\
                Let me go.\nWriting.\nOK.\nLet me make the calls.\nWriting now.\n\
                Let me go.\nOK.\nLet me write the calls.\nReading docs.\nLet me go.\n\
                OK writing.\nLet me make the calls.\nWriting.\nOK.\nLet me write.\n";
            peer.write_all(&Frame::Tokens(Tokens {
                req_id: submitted.req_id, ids: app.tok.encode_parallel(text, 1), logprobs: None, top: None,
            }).encode()).await.unwrap();
            done(&mut peer, &submitted, DoneReason::Length).await;
            let response = timeout(DEADLINE, request).await.unwrap().unwrap();
            assert_eq!(response.status(), StatusCode::OK, "tools={tools}, thinking={thinking}, enabled={enabled}");
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let json: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(json["choices"][0]["finish_reason"], "length");
            assert_eq!(app.sessions.lock().unwrap().len(), 1);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocked_terminal_callback_does_not_block_other_stream_tokens() {
        use crate::protocol::Tokens;
        use tokio::sync::oneshot;

        let path = SocketPath(std::env::temp_dir().join(format!(
            "pico-done-callback-{}.sock", std::process::id()
        )));
        let listener = UnixListener::bind(&path.0).unwrap();
        let engine = EngineClient::start(path.0.clone());
        let mut peer = connect(&listener, &engine).await;
        let request = Submit {
            req_id: 0,
            session_id: 1,
            base_session_id: 0,
            seed: 0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            max_new_tokens: 1,
            flags: 0,
            n_top_logprobs: 0,
            prefix_hint: 0,
            stop_ids: Vec::new(),
            prompt: vec![1],
            images: None,
            prediction: Vec::new(),
        };
        let (entered_tx, entered_rx) = oneshot::channel();
        let (unblock_tx, unblock_rx) = std::sync::mpsc::channel();
        let callback = move |reason| {
            let _ = entered_tx.send(reason);
            // The bound also lets the runtime shut down if the test fails before unblocking.
            let _ = unblock_rx.recv_timeout(DEADLINE * 2);
        };
        let (first_id, mut first_rx) = engine.submit_with_done(request.clone(), Some(Box::new(callback))).unwrap();
        let (second_id, mut second_rx) = engine.submit(Submit { session_id: 2, ..request }).unwrap();
        let first = submit(&mut peer).await;
        let second = submit(&mut peer).await;
        assert_eq!((first.req_id, second.req_id), (first_id, second_id));

        done(&mut peer, &first, DoneReason::Length).await;
        assert_eq!(timeout(DEADLINE, entered_rx).await.unwrap().unwrap(), DoneReason::Length);
        let token = Tokens { req_id: second_id, ids: vec![42], logprobs: None, top: None };
        peer.write_all(&Frame::Tokens(token.clone()).encode()).await.unwrap();
        let received = timeout(DEADLINE, second_rx.recv()).await;
        assert!(unblock_tx.send(()).is_ok(), "the first request's terminal callback blocked the second stream until its safety timeout");
        match received.expect("the first request's terminal callback blocked the second stream") {
            Some(EngineEvent::Tokens(t)) => assert_eq!(t, token),
            event => panic!("expected the second request's token, got {event:?}"),
        }

        assert!(matches!(
            timeout(DEADLINE, first_rx.recv()).await.unwrap(),
            Some(EngineEvent::Done(d)) if d.req_id == first_id && d.reason == DoneReason::Length
        ));
        done(&mut peer, &second, DoneReason::Length).await;
        assert!(matches!(
            timeout(DEADLINE, second_rx.recv()).await.unwrap(),
            Some(EngineEvent::Done(d)) if d.req_id == second_id
        ));
    }
}
