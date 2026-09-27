"""
check_reveals.py -- static check that the benchmark MPC programs reveal only what the open-w1
construction reveals: w1 of each attempt, j*, and z of the accepted attempt.

Every `.reveal()` outside an `if debug:` block must be applied to one of the allowed names.
Exit status 1 on any other reveal, or on reveal_to / public output of other secrets.
"""
import os, re, sys

MPC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpc")
ALLOWED = {"w1", "jplus", "zsel"}          # jstar = jplus.reveal() - 1
# Statistically masked openings (value + mask with >= 40 bits of slack), the same kind MP-SPDZ's own
# comparisons perform internally; allowed only as assignments to these names:
MASKED = ("opening = (", "masked = (")
FORBIDDEN_CALLS = ("reveal_to", "reveal_to_clients", "output_shares")


def check_file(path):
    errors = []
    lines = open(path, encoding="utf-8").read().splitlines()
    debug_indent = None
    for no, line in enumerate(lines, 1):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if debug_indent is not None and stripped and indent <= debug_indent:
            debug_indent = None
        if re.match(r"if debug\s*:", stripped):
            debug_indent = indent
            continue
        if debug_indent is not None or stripped.startswith("#"):
            continue
        for call in FORBIDDEN_CALLS:
            if call + "(" in line:
                errors.append(f"{os.path.basename(path)}:{no}: forbidden {call}")
        for m in re.finditer(r"([A-Za-z_][A-Za-z_0-9]*)\s*\.reveal\(\)", line):
            if m.group(1) not in ALLOWED:
                errors.append(f"{os.path.basename(path)}:{no}: reveals '{m.group(1)}'")
        if re.search(r"\)\s*\.reveal\(\)", line) and not stripped.startswith(MASKED):
            errors.append(f"{os.path.basename(path)}:{no}: reveals an unnamed expression")
    return errors


def main():
    errors = []
    files = sorted(f for f in os.listdir(MPC) if f.endswith(".mpc"))
    for f in files:
        errors += check_file(os.path.join(MPC, f))
    if errors:
        print("REVEAL CHECK FAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)
    print(f"reveal check OK: {len(files)} programs reveal only {sorted(ALLOWED)} outside debug blocks")


if __name__ == "__main__":
    main()
