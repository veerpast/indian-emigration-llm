#!/usr/bin/env python3
"""Collect authorized Reddit Q&A material through Reddit's official API.

Run this only after Reddit has approved the research use and the relevant
rightsholders have permitted model-training use. The explicit approval flag is
an intentional guardrail, not a claim that permission has been obtained.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import praw
from dotenv import load_dotenv
from tqdm import tqdm

REMOVED = {"[deleted]", "[removed]", ""}
WHITESPACE_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subreddits", required=True, help="comma-separated approved communities")
    parser.add_argument("--query", required=True, help="narrow research query")
    parser.add_argument("--limit", type=int, default=100, choices=range(1, 1001), metavar="1..1000")
    parser.add_argument("--min-score", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("data/raw/reddit_authorized.csv"))
    parser.add_argument(
        "--i-have-approval",
        action="store_true",
        help="confirm Reddit API approval and rightsholder permission for this use",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.i_have_approval:
        raise SystemExit(
            "Collection stopped. Obtain Reddit API approval and rightsholder permission, "
            "then re-run with --i-have-approval."
        )

    load_dotenv()
    required = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}")

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
        check_for_async=False,
    )

    communities = "+".join(part.strip() for part in args.subreddits.split(",") if part.strip())
    rows: list[dict[str, object]] = []
    submissions = reddit.subreddit(communities).search(
        args.query, sort="relevance", time_filter="all", limit=args.limit
    )

    for submission in tqdm(submissions, total=args.limit, desc="Approved API collection"):
        question = clean(submission.title)
        if not question or submission.score < args.min_score or submission.over_18:
            continue
        submission.comments.replace_more(limit=0)
        candidates = [
            comment
            for comment in submission.comments.list()
            if clean(getattr(comment, "body", "")) not in REMOVED
            and getattr(comment, "score", 0) >= args.min_score
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda comment: comment.score)
        rows.append(
            {
                "question": question,
                "answer": clean(best.body),
                "source": "Reddit Travel & Relocation Community Q&A",
                "submission_id": submission.id,
                "comment_id": best.id,
                "created_utc": int(submission.created_utc),
                "retrieved_via": "Reddit Data API",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "answer",
        "source",
        "submission_id",
        "comment_id",
        "created_utc",
        "retrieved_via",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} authorized Q&A rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

