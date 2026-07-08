"""
Corpus-aware regression evaluation harness.

Runs a fixed set of query/expected-substring test cases (eval/test_cases.json)
through the full LangGraph orchestration and reports pass/fail per case.

Unlike tests/ (corpus-independent unit tests, runnable anywhere with no
external services), this harness assumes the papers ingested during this
build are still in the Chroma store. Re-run this after any change to
retrieval, planning, or prompt logic — it's built directly from the same
query/answer pairs already verified by hand throughout this project, and
would have caught the Planner misrouting regression found earlier in this
session automatically instead of requiring a manual re-check.

Run: python scripts/run_eval.py
"""

import json
import sys

from researchmind.config import PROJECT_ROOT
from researchmind.graph import run_query

TEST_CASES_PATH = PROJECT_ROOT / "eval" / "test_cases.json"


def main() -> None:
    if not TEST_CASES_PATH.exists():
        print(f"No test cases found at {TEST_CASES_PATH}.")
        sys.exit(1)

    test_cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))

    passed = 0
    failed = 0

    for case in test_cases:
        name = case["name"]
        query = case["query"]
        expected = case["expect_contains"]

        try:
            result = run_query(query)
            answer = result["final_answer"].lower()
        except Exception as exc:
            print(f"❌ {name}: raised {type(exc).__name__}: {exc}")
            failed += 1
            continue

        missing = [e for e in expected if e.lower() not in answer]

        if not missing:
            print(f"✅ {name}")
            passed += 1
        else:
            print(f"❌ {name}: missing {missing} in answer: {result['final_answer'][:200]}")
            failed += 1

    print(f"\n--- Eval Summary ---")
    print(f"Passed: {passed} | Failed: {failed} | Total: {passed + failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()