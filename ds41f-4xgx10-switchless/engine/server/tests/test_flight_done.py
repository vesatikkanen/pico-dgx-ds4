#!/usr/bin/env python3
"""The flight's DONE, once, on both paths (test_flight_done.cpp).

serve.cpp reaches DONE two ways:

  * the single-flight path (decode_one): run() calls finish() alone, at the run's end;
  * the batch path (decode_batch): the loop's OnStop calls finish_early() inside the run, and finish()
    runs at the run's end.

Both must send it exactly once, and the reason the client was told must be the one the state ops use.
This extracts finish_early and finish's head verbatim and checks that.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def section(source, first, last):
    start = source.index(first)
    return source[start:source.index(last, start)]


def function_text(text):
    """A function as written: its signature, its statements, its closing brace."""
    lines = text.splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()
    if not lines[-1].strip() == "}":
        lines.append("  }")
    return "\n".join(lines) + "\n"


def body_only(text):
    """A function's statements, without its signature line or its closing brace."""
    lines = text.splitlines()[1:]
    while lines and lines[-1].strip() in ("", "}"):
        lines.pop()
    return "\n".join(lines) + "\n"


def check_finish_calls_early(source):
    """finish() must send the DONE itself, or the single-flight path sends none."""
    head = section(source, "  void finish(Flight& f, uint32_t why) {", "    const fe::SubmitHeader& h = f.rq.h;")
    lines = [line.strip() for line in head.splitlines()[1:] if line.strip()]
    if not lines or not lines[0].startswith("finish_early(f, why);"):
        raise SystemExit("finish() must call finish_early() first, or the single-flight path sends no DONE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serve", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "serve.cpp")
    args = parser.parse_args()
    source = args.serve.read_text()
    check_finish_calls_early(source)
    early = function_text(section(source, "  void finish_early(Flight& f, uint32_t why) {",
                                  "  // DONE (already sent when the flight ended inside the run)"))
    head = body_only(section(source, "  void finish(Flight& f, uint32_t why) {",
                             "    const fe::SubmitHeader& h = f.rq.h;"))
    with tempfile.TemporaryDirectory(prefix="pico-flight-done-") as temp:
        folder = Path(temp)
        (folder / "finish_early.inc").write_text(early)
        (folder / "finish_head.inc").write_text(head)
        fixture = Path(__file__).with_suffix(".cpp").resolve()
        binary = folder / "test_flight_done"
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
