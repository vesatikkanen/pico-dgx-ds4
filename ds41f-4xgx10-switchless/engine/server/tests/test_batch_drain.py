#!/usr/bin/env python3
"""Compile the decode loops' stop decisions with a synthetic entry boundary.

The batch drain (docs/speed-bench.md): a flight's DONE goes out the moment its entry ends, not when the
run's last flight does. serve.cpp learns of that end through OnStop, which the loops call from the one
place each decides an entry is done. This checks that contract on the real lambdas:

  * called once per entry, with the reason the loop recorded in end[k].why;
  * never for kEndPause or kEndSwitch, which are not ends (the request goes on);
  * not called for an entry that is still decoding;
  * a null OnStop is not a crash (main.cpp's loops pass none).
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def section(source, first, last):
    start = source.index(first)
    return source[start:source.index(last, start)]


def body(text):
    """The statements of an `auto x = [&](...) { ... };` lambda."""
    return text.split("{", 1)[1].rsplit("};", 1)[0]


# Every place an entry stops decoding must tell serve.cpp, or its DONE waits for the run's end (the drain).
# These are the sites that end an entry without an on_stop of their own, and why that is right:
COVERED = {
    # inside stop(): the on_stop call is the next statement
    "else live[k] = 0, end[k].fed = launched[k];",
    # inside finish(): the on_stop call is the next statement
    "live[k] = 0, ending[k] = 0, pend[k] = 0;",
    # the row queued behind an end: stop() already reported it, and finish_early is idempotent
    "live[k] = 0, ending[k] = 0, x.fed = launched[k];",
}


def check_ends(source):
    """Every `live[...] = 0` either reports the end or is one of the known-covered sites."""
    missing = []
    for number, line in enumerate(source.splitlines(), 1):
        text = line.strip()
        if not re.search(r"live\[[^\]]*\]\s*=\s*0", text):
            continue
        if text in COVERED:
            continue
        # a pre-loop guard: it never decodes, so it reports the end itself
        window = "\n".join(source.splitlines()[number - 1 : number + 2])
        if "on_stop" not in window:
            missing.append(f"cmd.cpp:{number}: {text}")
    if missing:
        raise SystemExit(
            "an entry ends without reporting it (its DONE would wait for the run's end):\n  "
            + "\n  ".join(missing)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "cmd.cpp")
    args = parser.parse_args()
    source = args.cmd.read_text()
    check_ends(source)
    # the two lambdas that decide an entry's end, verbatim
    batch_stop = section(source, "  auto stop = [&](int k, uint32_t why) {", "  launch();\n  launch();")
    spec_finish = section(source, "  auto finish = [&](int k, int kv_end) {", "  struct Step {")
    with tempfile.TemporaryDirectory(prefix="pico-batch-drain-") as temp:
        folder = Path(temp)
        (folder / "batch_stop.inc").write_text(body(batch_stop))
        (folder / "spec_finish.inc").write_text(body(spec_finish))
        fixture = Path(__file__).with_suffix(".cpp").resolve()
        binary = folder / "test_batch_drain"
        subprocess.run(
            [
                "nice", "-n", "19", os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                "-Werror", "-fsanitize=undefined", "-I", temp, str(fixture), "-o", str(binary),
            ],
            check=True,
        )
        subprocess.run(["nice", "-n", "19", str(binary)], check=True)


if __name__ == "__main__":
    main()
