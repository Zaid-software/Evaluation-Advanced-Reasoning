import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRATEGIES_DIR = os.path.join(BASE_DIR, "strategies")

LEAK_PATTERNS = [
    re.compile(r"problem\[[\'\"]answer[\'\"]\]"),
    re.compile(r"problem\.get\([\'\"]answer[\'\"]"),
    re.compile(r"\.answer\b"),
]


ALLOWLISTED_FILES = {"base.py"}


def check_file(path):
    violations = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in LEAK_PATTERNS:
                if pattern.search(line):
                    violations.append((lineno, line.strip()))
    return violations


def main():
    any_violations = False
    for fname in sorted(os.listdir(STRATEGIES_DIR)):
        if not fname.endswith(".py") or fname in ALLOWLISTED_FILES or fname == "__init__.py":
            continue
        path = os.path.join(STRATEGIES_DIR, fname)
        violations = check_file(path)
        if violations:
            any_violations = True
            print(f"[LEAK RISK] {fname}:")
            for lineno, line in violations:
                print(f"    line {lineno}: {line}")
        else:
            print(f"[OK] {fname}: no ground-truth access detected")

    if any_violations:
        print("\nFAIL: potential ground-truth leakage detected in strategy code.")
        sys.exit(1)
    else:
        print("\nPASS: no strategy file reads problem['answer'].")


if __name__ == "__main__":
    main()
