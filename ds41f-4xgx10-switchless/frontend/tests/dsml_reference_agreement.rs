//! Differential checks against the strict reference parser (`parse_message_from_completion_text`,
//! the port of `encoding.py`'s `parse_message_from_completion_text`).
//!
//! The dangerous direction is one-way: a reply the reference reads but the streamer quarantines is
//! answered with a 500, and a client retries that request — the "loop after a failing tool call"
//! symptom. The other direction (the reference raises, the streamer still finds a call) is by
//! design: the streamer is the lenient one that serves.

use dsfront::encoding::{
    parse_message_from_completion_text, Delta, Encoding, ParsedMessage, StreamParser, ThinkingMode, THINK_END,
};

const D: &str = "\u{ff5c}DSML\u{ff5c}";

fn call(name: &str, params: &[(&str, &str, &str)]) -> String {
    let mut s = format!("<{D} invoke name=\"{name}\">\n");
    for (k, flag, v) in params {
        s.push_str(&format!("<{D} parameter name=\"{k}\" string=\"{flag}\">{v}</{D} parameter>\n"));
    }
    s.push_str(&format!("</{D} invoke>\n"));
    s
}

fn block(calls: &[String]) -> String {
    format!("\n\n<{D} calls>\n{}</{D} calls>", calls.concat())
}

fn stream(text: &str, n: usize, stream_args: bool, mode: ThinkingMode) -> (Vec<Delta>, StreamParser) {
    let mut p = StreamParser::new(mode).stream_args(stream_args);
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    for ch in chars.chunks(n) {
        p.push(&ch.iter().collect::<String>(), &mut out);
    }
    p.finish(&mut out);
    (out, p)
}

fn message(deltas: Vec<Delta>) -> ParsedMessage {
    let mut m = ParsedMessage::default();
    deltas.into_iter().for_each(|d| m.add(d));
    m
}

/// Wherever the reference reads the reply, the streamer must read it the same way, at every chunk
/// width and in both tool-call modes.
fn agrees(name: &str, text: &str, mode: ThinkingMode) {
    let Ok(want) = parse_message_from_completion_text(text, mode, Encoding::V41) else { return };
    for n in 1..=text.chars().count().min(20) {
        for stream_args in [false, true] {
            let (out, p) = stream(text, n, stream_args, mode);
            assert_eq!(
                p.protocol_error(),
                None,
                "{name}: {n} chars at a time, pieces={stream_args}: the reference reads {} call(s), \
                 the streamer quarantines the reply",
                want.tool_calls.len()
            );
            assert_eq!(
                message(out).tool_calls,
                want.tool_calls,
                "{name}: {n} chars at a time, pieces={stream_args}"
            );
        }
    }
}

#[test]
fn shapes_a_model_writes() {
    let chat = ThinkingMode::Chat;
    let think = ThinkingMode::Thinking;

    agrees("plain", &block(&[call("write", &[("path", "true", "a.txt")])]), chat);
    agrees("no parameters", &block(&[call("now", &[])]), chat);
    agrees("two calls", &block(&[call("a", &[("x", "true", "1")]), call("b", &[("y", "false", "2")])]), chat);
    agrees("three calls", &block(&[call("a", &[]), call("b", &[]), call("c", &[])]), chat);
    agrees("thinking then a call", &format!("reasoning{THINK_END}answer{}", block(&[call("a", &[])])), think);

    agrees("value with a quote and a backslash", &block(&[call("w", &[("t", "true", "a \"b\" \\c")])]), chat);
    agrees("value with a newline", &block(&[call("w", &[("t", "true", "line1\nline2\n")])]), chat);
    agrees("value with CRLF", &block(&[call("w", &[("t", "true", "a\r\nb")])]), chat);
    agrees("value with a DSML opener", &block(&[call("w", &[("t", "true", &format!("<{D} invoke name=\"x\">"))])]), chat);
    agrees("value with a bare '<'", &block(&[call("w", &[("t", "true", "a < b")])]), chat);
    agrees("value that is only '<'", &block(&[call("w", &[("t", "true", "<")])]), chat);
    agrees("value with a JSON object", &block(&[call("w", &[("o", "false", "{\"n\": 3, \"t\": [\"<x>\"]}")])]), chat);
    agrees("value with a JSON array", &block(&[call("w", &[("o", "false", "[1, 2, 3]")])]), chat);
    agrees("value with a JSON string", &block(&[call("w", &[("o", "false", "\"s\"")])]), chat);
    agrees("value with a JSON null", &block(&[call("w", &[("o", "false", "null")])]), chat);
    agrees("value with a bare word", &block(&[call("w", &[("o", "false", "pwd")])]), chat);
    agrees("value with a capitalised boolean", &block(&[call("w", &[("o", "false", "True")])]), chat);
    agrees("value with trailing whitespace", &block(&[call("w", &[("o", "false", "3 ")])]), chat);
    agrees("empty value", &block(&[call("w", &[("t", "true", "")])]), chat);
    agrees("empty JSON value", &block(&[call("w", &[("o", "false", "")])]), chat);
    agrees("unicode value", &block(&[call("w", &[("t", "true", "こんにちは é 🎉")])]), chat);
    agrees("prose with a newline before", &format!("before\n\n{}", block(&[call("a", &[])])), chat);
}

/// A `string="false"` value the model wrote as a bare word. `encoding.py`'s `parse_tool_calls` does
/// not validate it, so the streamer must not turn the reply into a 500.
#[test]
fn bare_non_json_value_is_not_a_protocol_error() {
    let text = block(&[call("run", &[("cmd", "false", "pwd")])]);
    let want = parse_message_from_completion_text(&text, ThinkingMode::Chat, Encoding::V41)
        .expect("the reference parser accepts a bare value");
    assert_eq!(want.tool_calls, [("run".to_string(), "{\"cmd\": pwd}".to_string())]);
    agrees("bare non-JSON value", &text, ThinkingMode::Chat);
}

/// The no-space tag spelling (`</｜DSML｜parameter>`) must not leak into the streamed argument pieces.
#[test]
fn tight_tag_spelling_streams_clean_arguments() {
    let text = block(&[call("write", &[("path", "true", "a.txt"), ("text", "true", "hello")])])
        .replace(&format!("{D} "), D);
    let want = "{\"path\": \"a.txt\", \"text\": \"hello\"}";
    for n in 1..=text.chars().count() {
        let (out, p) = stream(&text, n, true, ThinkingMode::Chat);
        assert_eq!(p.protocol_error(), None, "{n} at a time");
        let got = message(out).tool_calls[0].1.clone();
        assert_eq!(got, want, "{n} at a time");
        assert!(
            serde_json::from_str::<serde_json::Value>(&got).is_ok(),
            "{n} at a time: {got} is not JSON"
        );
    }
}
