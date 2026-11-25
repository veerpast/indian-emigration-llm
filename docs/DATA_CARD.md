# Dataset card: Indian Emigration Q&A Corpus

## Summary

The local research corpus contains **30,324** question-answer pairs scraped from
Reddit, then cleaned and pre-processed into three columns: `question`, `answer`,
and `source`. Despite its name, `source` is a derived category label rather than
an acquisition-origin field. The corpus covers consular interviews, student
visas, immigration regulation, relocation advice, and related topics. The file
is used for supervised fine-tuning and offline evaluation experiments.

The full CSV is intentionally not distributed in the public repository until
source-level rights, consent, privacy, and takedown procedures are documented.
This is especially important for user-generated content.

## Local snapshot (audited 25 November 2026)

| Post-processing category label | Rows | Share |
|---|---:|---:|
| Indian Student Visa Consular Interview Experiences | 17,934 | 59.14% |
| Complete Consular Visa Interview Debriefs | 7,813 | 25.77% |
| Immigration Legal & Regulatory QA (USCIS/DOS) | 2,894 | 9.54% |
| Reddit Travel & Relocation Community Q&A | 742 | 2.45% |
| Study Abroad, Accommodation & Visa Advisory Corpus | 693 | 2.29% |
| F-1 Student Visa Interview Records | 164 | 0.54% |
| Consular Visa Interview Guidelines | 84 | 0.28% |

Structural audit results:

- 30,324 non-empty questions; all question strings are unique.
- 30,258 unique answer strings (66 repeated answer occurrences).
- Mean question length: 88.9 characters.
- Mean answer length: 375.0 characters.
- Longest question: 618 characters.
- Longest answer: 5,630 characters.
- Possible phone-pattern matches: 9 rows (manual review required).
- Possible email-pattern matches: 0 rows.

Phone-pattern matches are heuristic flags, not confirmed identifiers. Review the
candidate rows locally and redact or remove any direct identifiers before
training, demonstration, or release. The raw rows are not reproduced here.

These measurements establish structure, not truthfulness or legal accuracy.

## Intended uses

- Academic experiments in domain adaptation with LoRA.
- Controlled comparison of open-source instruction models.
- Offline research into quantization and resource-constrained inference.
- Demonstrations that include a clear non-legal-advice notice.

## Out-of-scope uses

- Legal, immigration, or consular decision-making.
- Automated eligibility decisions or risk scoring.
- Production advice without retrieval from current official sources and human
  review.
- Identity inference, profiling, surveillance, or contacting source authors.
- Commercial use of user-generated content without the necessary rights.

## Collection and provenance

The present CSV records a derived category but does not contain a complete
row-level Reddit provenance ledger. That is a limitation. Before releasing a
dataset version, create a manifest for every source item/thread with:

- source URL or document identifier;
- access date and collection method;
- license or written permission;
- allowed research and redistribution purposes;
- deletion/takedown status;
- transformation and de-identification steps;
- content hash and dataset version.

The repository includes `scripts/reddit_collector.py` as an example of API-based
collection for **approved** research only. As of 25 November 2026, Reddit's
Responsible Builder Policy requires explicit API approval, and its Data API
Terms restrict using user content for AI training without rightsholder
permission. The script therefore refuses to run unless the operator explicitly
confirms approval. This guardrail does not itself grant or prove permission.

## Processing recommendations

1. Normalize Unicode and whitespace without changing substantive meaning.
2. Remove deleted/removed content and rows without usable answers.
3. Detect direct identifiers (email, phone, address, passport/application IDs).
4. Review detected rows manually; redact or remove them.
5. Deduplicate before splitting to reduce train/test leakage.
6. Split by source thread or document, not only by row, where grouping metadata
   exists.
7. Freeze a test set before tuning hyperparameters.
8. Record all transformations and hashes in a versioned manifest.

Run the structural audit with:

```bash
python3 scripts/validate_dataset.py /path/to/indian_emigration_qa_purified.csv --json
```

## Known limitations

- Answers may be anecdotal, outdated, incomplete, or jurisdiction-specific.
- Source labels are broad and do not substitute for row-level citations.
- The corpus is skewed toward student visa interview material.
- It may encode demographic, geographic, socioeconomic, or language biases.
- Regulatory guidance changes; model weights do not update automatically.
- Fine-tuned models can hallucinate even when trained on accurate examples.

## Release checklist

- [ ] Row-level provenance and license manifest completed.
- [ ] Reddit/API approval and relevant rightsholder permissions recorded.
- [ ] Direct-identifier scan and manual review completed.
- [ ] Takedown contact and deletion workflow published.
- [ ] Train/validation/test split hashes published.
- [ ] Dataset license reviewed by the university or project supervisor.

