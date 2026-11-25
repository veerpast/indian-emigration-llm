import pytest

from scripts.reddit_collector import ensure_output_available


def test_collector_refuses_to_overwrite_existing_output(tmp_path):
    output = tmp_path / "existing.csv"
    output.write_text("preserve this", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        ensure_output_available(output, overwrite=False)

    assert output.read_text(encoding="utf-8") == "preserve this"


def test_collector_allows_explicit_overwrite(tmp_path):
    output = tmp_path / "existing.csv"
    output.write_text("replaceable", encoding="utf-8")

    ensure_output_available(output, overwrite=True)
