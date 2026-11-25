#!/usr/bin/env python3
"""Validate and summarize the project's question-answer CSV.

This script is deliberately dependency-free so it can run in CI and on Colab.
It reports structural quality signals; it does not certify factual correctness,
privacy compliance, or permission to redistribute source content.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

REQUIRED_COLUMNS = ("question", "answer", "source")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){9,14}(?!\d)")


def audit(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")

        row_count = 0
        empty = Counter()
        source_counts = Counter()
        questions = Counter()
        answers = Counter()
        possible_emails = 0
        possible_phones = 0
        question_chars = 0
        answer_chars = 0

        for row in reader:
            row_count += 1
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            source = (row.get("source") or "").strip()
            for field, value in (("question", question), ("answer", answer), ("source", source)):
                if not value:
                    empty[field] += 1
            questions[question] += 1
            answers[answer] += 1
            source_counts[source] += 1
            question_chars += len(question)
            answer_chars += len(answer)
            joined = f"{question}\n{answer}"
            possible_emails += bool(EMAIL_RE.search(joined))
            possible_phones += bool(PHONE_RE.search(joined))

    return {
        "file": str(path),
        "columns": list(columns),
        "rows": row_count,
        "empty_fields": dict(empty),
        "duplicate_questions": sum(count - 1 for count in questions.values() if count > 1),
        "duplicate_answers": sum(count - 1 for count in answers.values() if count > 1),
        "mean_question_chars": round(question_chars / row_count, 2) if row_count else 0,
        "mean_answer_chars": round(answer_chars / row_count, 2) if row_count else 0,
        "possible_email_rows": possible_emails,
        "possible_phone_rows": possible_phones,
        "source_counts": dict(source_counts.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--strict", action="store_true", help="fail on empty fields or duplicate questions")
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(f"file not found: {args.csv_path}")

    try:
        report = audit(args.csv_path)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"dataset validation failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Rows: {report['rows']:,}")
        print(f"Columns: {', '.join(report['columns'])}")
        print(f"Empty fields: {report['empty_fields'] or 'none'}")
        print(f"Duplicate questions: {report['duplicate_questions']:,}")
        print(f"Duplicate answers: {report['duplicate_answers']:,}")
        print(f"Possible email rows (manual review): {report['possible_email_rows']:,}")
        print(f"Possible phone rows (manual review): {report['possible_phone_rows']:,}")
        print("Sources:")
        for source, count in report["source_counts"].items():
            print(f"  {count:>6,}  {source}")

    if args.strict and (report["empty_fields"] or report["duplicate_questions"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

