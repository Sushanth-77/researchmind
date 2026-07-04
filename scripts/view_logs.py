"""
Summarize structured Groq call logs: totals and per-caller breakdown.

Run: python scripts/view_logs.py
"""

import json
from collections import defaultdict

from researchmind.observability import LOG_FILE


def main() -> None:
    if not LOG_FILE.exists():
        print(f"No log file found at {LOG_FILE}. Run some queries first.")
        return

    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("Log file exists but is empty.")
        return

    total_calls = len(entries)
    errors = [e for e in entries if e["error"] is not None]
    successful = [e for e in entries if e["error"] is None]

    total_tokens = sum(e["total_tokens"] or 0 for e in successful)
    total_latency = sum(e["latency_seconds"] for e in entries)

    print(f"--- Groq Call Log Summary ---")
    print(f"Total calls: {total_calls}")
    print(f"Successful: {len(successful)} | Errors: {len(errors)}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Total latency: {total_latency:.2f}s")
    print(f"Total cost: $0.00 (Groq free tier)")

    print(f"\n--- Breakdown by caller ---")
    by_caller = defaultdict(lambda: {"calls": 0, "tokens": 0, "latency": 0.0})
    for e in entries:
        stats = by_caller[e["caller"]]
        stats["calls"] += 1
        stats["tokens"] += e["total_tokens"] or 0
        stats["latency"] += e["latency_seconds"]

    for caller, stats in sorted(by_caller.items()):
        avg_latency = stats["latency"] / stats["calls"]
        print(
            f"  {caller}: {stats['calls']} calls, "
            f"{stats['tokens']} tokens, avg {avg_latency:.2f}s/call"
        )

    if errors:
        print(f"\n--- Errors ---")
        for e in errors:
            print(f"  {e['caller']}: {e['error']}")


if __name__ == "__main__":
    main()