//! Stop strings and the repetition-loop guard, applied to the generated stream.

/// Finds the client's stop strings in streamed text, holding back a tail that could be the start
/// of one. The stop string itself is not passed on.
pub struct StopMatcher {
    stops: Vec<String>,
    buf: String,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        StopMatcher { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), buf: String::new() }
    }

    /// Text that can be passed on, and whether a stop string ended it.
    pub fn push(&mut self, s: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (s.to_string(), false);
        }
        self.buf.push_str(s);
        if let Some(p) = self.stops.iter().filter_map(|st| self.buf.find(st.as_str())).min() {
            let out = self.buf[..p].to_string();
            self.buf.clear();
            return (out, true);
        }
        let mut keep = 0;
        for st in &self.stops {
            for k in (1..st.len().min(self.buf.len() + 1)).rev() {
                let at = self.buf.len() - k;
                if self.buf.is_char_boundary(at) && st.starts_with(&self.buf[at..]) {
                    keep = keep.max(k);
                    break;
                }
            }
        }
        let n = self.buf.len() - keep;
        let out = self.buf[..n].to_string();
        self.buf.drain(..n);
        (out, false)
    }

    /// End of the stream: the held-back tail is ordinary text after all.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }
}

/// The kit's loop abort (adapter/loop_abort.py), same defaults: stop when the last `repeats`
/// copies of an n-gram of `ngram..=ngram_max` tokens are identical, when one token repeats
/// `identical` times, or when the last `line_repeats` non-empty lines (at least `line_min` chars)
/// of the last 512 tokens' text are identical.
#[derive(Clone, Debug)]
pub struct LoopGuard {
    pub ngram: usize,
    pub ngram_max: usize,
    pub repeats: usize,
    pub identical: usize,
    pub line_repeats: usize,
    pub line_min: usize,
    /// A run of this many non-empty lines, each shorter than `line_min` and drawn from at most
    /// `short_distinct` different texts, is a loop too. A model that stalls in its reasoning writes
    /// short lines from a handful of phrases ("OK.", "Let me write.", "Writing.", "Go.") in no fixed
    /// order, so neither the n-gram rule (no exact period) nor the line rule (every line is below
    /// `line_min`) sees it, and it runs to `max_tokens` without ever reaching a tool call.
    pub short_lines: usize,
    pub short_distinct: usize,
}

pub const LINE_WINDOW_TOKENS: usize = 512;

/// A line of reasoning, and whether it says anything: a line at least `SUBSTANTIVE` characters long
/// is the model reporting a finding, not stalling.
struct Line {
    text: String,
    substantive: bool,
}

/// Detect a stall in reasoning: a run of short lines that only announce an intention to act, with
/// nothing said in between. A model that stalls writes "Let me read the docs.", "OK.", "Writing."
/// and the like for the rest of the budget and never emits the call it keeps announcing, so neither
/// the n-gram rule (no exact period) nor the line rule (no line reaches `line_min`) sees it.
///
/// Only reasoning is examined, and only for a request that offers tools; a tool call resets the
/// buffer, because a call is the progress the stall never makes. A bounded line buffer makes the
/// result independent of token and engine batch boundaries.
#[derive(Default)]
pub struct ReasoningStall {
    partial: String,
    oversized: bool,
    lines: std::collections::VecDeque<Line>,
}

/// The verbs a line that announces an action starts with, base form and gerund.
const ACTION_VERBS: [&str; 98] = [
    "read", "reads", "reading", "write", "writes", "writing", "make", "makes", "making", "call",
    "calls", "calling", "go", "going", "do", "does", "doing", "check", "checks", "checking", "look",
    "looks", "looking", "verify", "verifies", "verifying", "run", "runs", "running", "fix", "fixes",
    "fixing", "implement", "implements", "implementing", "deploy", "deploys", "deploying", "grep",
    "greps", "grepping", "search", "searches", "searching", "view", "views", "viewing", "create",
    "creates", "creating", "update", "updates", "updating", "add", "adds", "adding", "edit", "edits",
    "editing", "test", "tests", "testing", "build", "builds", "building", "commit", "commits",
    "committing", "push", "pushes", "pushing", "pull", "pulls", "pulling", "install", "installs",
    "installing", "enable", "enables", "enabling", "configure", "configures", "configuring",
    "scaffold", "scaffolds", "scaffolding", "paste", "pastes", "pasting", "ask", "asks", "asking",
    "explain", "explains", "explaining", "present", "presents", "presenting",
];

/// Words a line may open with before the verb: "let me read", "I'll write", "OK, writing".
const LEADS: [&str; 16] = [
    "let", "i'll", "ill", "i", "we", "we'll", "well", "now", "ok", "okay", "so", "then", "and",
    "but", "just", "first",
];

/// Words that may sit between a lead and the verb: "let me just do it".
const FILLERS: [&str; 8] = ["me", "us", "just", "now", "also", "then", "really", "actually"];

/// Lines that announce nothing but are part of the same stall.
const FILLER_LINES: [&str; 24] = [
    "ok", "done", "yes", "no", "now", "enough", "final", "actually", "hmm", "wait", "right", "so",
    "and", "but", "then", "next", "first", "finally", "good", "great", "fine", "sure", "alright",
    "okay",
];

/// Whether a line only announces an action ("Let me read the docs.", "Writing.", "OK.").
fn is_action(line: &str) -> bool {
    let words: Vec<String> = line
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_ascii_lowercase())
        .filter(|w| !w.is_empty())
        .collect();
    let Some(first) = words.first() else { return false };
    if words.len() == 1 && FILLER_LINES.contains(&first.as_str()) {
        return true;
    }
    if ACTION_VERBS.contains(&first.as_str()) {
        return true;
    }
    if !LEADS.contains(&first.as_str()) {
        return false;
    }
    // "let me just do it": skip the fillers between the lead and the verb.
    words[1..].iter().find(|w| !FILLERS.contains(&w.as_str())).is_some_and(|w| ACTION_VERBS.contains(&w.as_str()))
}

impl ReasoningStall {
    /// Lines kept: the window the rule looks at.
    const WINDOW: usize = 16;
    /// A line this long says something, so the window is not a stall.
    const SUBSTANTIVE: usize = 40;
    /// A line longer than this is not buffered: it is substantive anyway.
    const MAX_LINE_BYTES: usize = 512;
    /// How many of the window's lines must announce an action.
    const ACTIONS: usize = 12;

    /// A tool call: the progress a stall never makes, so what came before it is not a stall.
    pub fn reset(&mut self) {
        self.partial.clear();
        self.oversized = false;
        self.lines.clear();
    }

    pub fn push(&mut self, text: &str) -> bool {
        for c in text.chars() {
            if c == '\n' {
                let line = self.partial.trim();
                let substantive = self.oversized || line.chars().count() >= Self::SUBSTANTIVE;
                if substantive || !line.is_empty() {
                    self.lines.push_back(Line { text: line.to_owned(), substantive });
                    if self.lines.len() > Self::WINDOW {
                        self.lines.pop_front();
                    }
                }
                self.partial.clear();
                self.oversized = false;
                if self.lines.len() == Self::WINDOW && self.is_stalled() {
                    return true;
                }
            } else if !self.oversized {
                if self.partial.len() + c.len_utf8() > Self::MAX_LINE_BYTES {
                    self.partial.clear();
                    self.oversized = true;
                } else {
                    self.partial.push(c);
                }
            }
        }
        false
    }

    fn is_stalled(&self) -> bool {
        if self.lines.iter().any(|l| l.substantive) {
            return false;
        }
        self.lines.iter().filter(|l| is_action(&l.text)).count() >= Self::ACTIONS
    }
}

impl Default for LoopGuard {
    fn default() -> Self {
        LoopGuard {
            ngram: 32,
            ngram_max: 256,
            repeats: 4,
            identical: 64,
            line_repeats: 8,
            line_min: 16,
            short_lines: 24,
            short_distinct: 6,
        }
    }
}

/// Python's str.splitlines boundaries.
fn is_line_break(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

fn py_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\x1c'..='\x1f')
}

impl LoopGuard {
    /// The number of tokens to keep if the output is looping (one copy of the cycle, or one token
    /// of a run; all of it for repeated lines), else None. `tail_text` is the text of the last
    /// LINE_WINDOW_TOKENS ids.
    pub fn check(&self, ids: &[u32], tail_text: &str) -> Option<usize> {
        let n = ids.len();
        if n == 0 {
            return None;
        }
        if self.identical > 0 && n >= self.identical {
            let last = ids[n - 1];
            let run = ids.iter().rev().take_while(|&&x| x == last).count();
            if run >= self.identical {
                return Some(n - run + 1);
            }
        }
        if self.ngram > 0 && self.repeats > 0 {
            let max_p = self.ngram_max.max(self.ngram).min(n / self.repeats);
            for period in self.ngram..=max_p {
                let unit = &ids[n - period..];
                if (1..self.repeats).all(|r| &ids[n - (r + 1) * period..n - r * period] == unit) {
                    return Some(n - (self.repeats - 1) * period);
                }
            }
        }
        if self.line_repeats > 0 && self.line_min > 0 {
            let lines: Vec<&str> =
                tail_text.split(is_line_break).filter(|l| !l.trim_matches(py_space).is_empty()).map(|l| l.trim_end_matches(py_space)).collect();
            if lines.len() >= self.line_repeats {
                let last = lines[lines.len() - 1];
                if last.chars().count() >= self.line_min && lines[lines.len() - self.line_repeats..].iter().all(|l| *l == last) {
                    return Some(n);
                }
            }
            // The same stall with no fixed order and no line long enough for the rule above.
            if self.short_lines > 0 && self.short_distinct > 0 && lines.len() >= self.short_lines {
                let run = &lines[lines.len() - self.short_lines..];
                if run.iter().all(|l| l.chars().count() < self.line_min) {
                    let mut seen: Vec<&str> = Vec::new();
                    for l in run {
                        if !seen.contains(l) {
                            seen.push(l);
                            if seen.len() > self.short_distinct {
                                break;
                            }
                        }
                    }
                    if seen.len() <= self.short_distinct {
                        return Some(n);
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_strings() {
        let mut m = StopMatcher::new(vec!["STOP".into(), "\n\n".into()]);
        let (a, s1) = m.push("hello ST");
        assert_eq!((a.as_str(), s1), ("hello ", false));
        let (b, s2) = m.push("ORM");
        assert_eq!((b.as_str(), s2), ("STORM", false));
        let (c, s3) = m.push("x\n");
        assert_eq!((c.as_str(), s3), ("x", false));
        let (d, s4) = m.push("\nrest");
        assert_eq!((d.as_str(), s4), ("", true));
    }

    #[test]
    fn loops() {
        let g = LoopGuard::default();
        let cycle: Vec<u32> = (0..6).collect();
        let ids: Vec<u32> = cycle.iter().cycle().take(36 * 4).cloned().collect();
        assert_eq!(g.check(&ids, ""), Some(36));
        assert_eq!(g.check(&ids[..36 * 4 - 1], ""), None);
        assert_eq!(g.check(&vec![7; 64], ""), Some(1));
        let text = "line that repeats itself\n".repeat(8);
        assert_eq!(g.check(&[1, 2, 3], &text), Some(3));
        assert_eq!(g.check(&[1, 2, 3], &"short\n".repeat(8)), None);
    }

    /// A stall in reasoning: short lines from a handful of phrases, in no fixed order, so neither
    /// the n-gram rule (no exact period) nor the line rule (no line reaches line_min) fires.
    #[test]
    fn short_line_stall() {
        let g = LoopGuard::default();
        let stall = "OK.\nLet me write.\nWriting.\nGo.\nOK.\nLet me write.\nWriting.\nLet me write.\nOK.\nWriting.\n\
                     Let me write.\nGo.\nWriting.\nOK.\nLet me write.\nWriting.\nGo.\nOK.\nLet me write.\nWriting.\n\
                     Let me write.\nOK.\nWriting.\nLet me write.\nGo.\nWriting.\nOK.\nLet me write.\nWriting.\n";
        assert_eq!(g.check(&[1, 2, 3], stall), Some(3));
        // the same lines, but each long enough for the line rule to have its own say
        let long = "a line that is long enough\n".repeat(24);
        assert_eq!(g.check(&[1, 2, 3], &long), Some(3));
        // prose of short lines is not a stall: too many different ones
        let prose = (0..24).map(|i| format!("line {i}\n")).collect::<String>();
        assert_eq!(g.check(&[1, 2, 3], &prose), None);
        // a stall that has not run long enough yet
        assert_eq!(g.check(&[1, 2, 3], "OK.\nWriting.\nOK.\nWriting.\n"), None);
    }

    #[test]
    fn reasoning_intentions_across_chunks() {
        let cycle = ["Let me read the docs.", "OK.", "Let me write the calls.",
            "Making calls.", "Let me go.", "Writing now."];
        let text = (0..24).map(|i| format!("{}\n\n", cycle[i % cycle.len()])).collect::<String>();
        let mut g = ReasoningStall::default();
        // one character at a time: the result does not depend on where the chunks fall
        let mut fired = None;
        for (i, c) in text.char_indices() {
            if g.push(&c.to_string()) {
                fired = Some(i);
                break;
            }
        }
        assert!(fired.is_some(), "a stall fed one character at a time was not seen");
        // the same text as one chunk fires at the same place
        let mut whole = ReasoningStall::default();
        assert!(whole.push(&text));
        // prose with no line breaks is not a stall
        assert!(!ReasoningStall::default().push(&text.replace('\n', " ")));
    }

    #[test]
    fn reasoning_intentions_match_reported_transcript() {
        // The tail of the reported transcript: the model announces the same calls over and over.
        let text = "Making calls.\nLet me write.\nOK.\nLet me just do it.\nReading docs.\n\
                    Let me write the calls.\nOK.\nLet me write.\nI'll read the two docs.\n\
                    Let me go.\nWriting.\nOK.\nLet me make the calls.\nWriting now.\n\
                    Let me go.\nOK.\nLet me write the calls.\nReading docs.\nLet me go.\n\
                    OK writing.\nLet me make the calls.\nWriting.\nOK.\nLet me write.\n";
        assert!(ReasoningStall::default().push(text));
    }

    #[test]
    fn reasoning_intentions_are_conservative() {
        // Two phrases alternating is a stall too: nothing is said between them.
        assert!(ReasoningStall::default().push(&"OK.\nWriting.\n".repeat(12)));
        // A line that says something ends the stall: the model is reporting a finding.
        let finding = "The manifest lists three services and one function binding for the module.\n";
        assert!(!ReasoningStall::default().push(&format!("{}{}", "Let me read.\n".repeat(8), finding)));
        // Distinct subjects are work, not a stall.
        assert!(!ReasoningStall::default().push(
            &(0..24).map(|i| format!("Let me inspect item {i}.\n")).collect::<String>()));
        // Code and lists are not reasoning about actions.
        assert!(!ReasoningStall::default().push(&"    return value;\n}\n".repeat(12)));
        assert!(!ReasoningStall::default().push(
            &(1..25).map(|i| format!("{i}. Install the dependency and restart the service.\n")).collect::<String>()));
        // A tool call is progress: it clears what came before, so two short runs either side of one
        // do not add up to a stall.
        let mut g = ReasoningStall::default();
        assert!(!g.push(&"Let me read.\nOK.\n".repeat(4)));
        g.reset();
        assert!(!g.push(&"Let me read.\nOK.\n".repeat(4)));
        // Without the reset the same two runs are one stall.
        assert!(ReasoningStall::default().push(&"Let me read.\nOK.\n".repeat(8)));
        // A very long line is substantive even though it is not buffered.
        let mut g = ReasoningStall::default();
        assert!(!g.push(&format!("{}{}{}", "Let me read.\n".repeat(8), format!("{}\n", "x".repeat(513)),
            "Let me read.\n".repeat(8))));
    }
}
