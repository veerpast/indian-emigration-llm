.PHONY: validate test serve

validate:
	python3 scripts/validate_dataset.py indian_emigration_qa_purified.csv

test:
	pytest -q

serve:
	python3 -m http.server 8000 --directory dist

