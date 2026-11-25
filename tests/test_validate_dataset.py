import csv

from scripts.validate_dataset import audit


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("question", "answer", "source"))
        writer.writeheader()
        writer.writerows(rows)


def test_audit_counts_rows_sources_and_duplicates(tmp_path):
    csv_path = tmp_path / "sample.csv"
    write_csv(
        csv_path,
        [
            {"question": "What is ECR?", "answer": "A passport category.", "source": "A"},
            {"question": "What is ECR?", "answer": "A passport category.", "source": "A"},
            {"question": "What is ECNR?", "answer": "A passport status.", "source": "B"},
        ],
    )

    report = audit(csv_path)

    assert report["rows"] == 3
    assert report["duplicate_questions"] == 1
    assert report["duplicate_answers"] == 1
    assert report["source_counts"] == {"A": 2, "B": 1}


def test_audit_flags_possible_contact_details(tmp_path):
    csv_path = tmp_path / "contact.csv"
    write_csv(
        csv_path,
        [
            {
                "question": "Who can I contact?",
                "answer": "Email student@example.com or call +91 98765 43210.",
                "source": "Example",
            }
        ],
    )

    report = audit(csv_path)

    assert report["possible_email_rows"] == 1
    assert report["possible_phone_rows"] == 1

