# Parked: synthetic insurance appetite set

Kept for the document-pipeline series (see PLAN.md, section 10). It isn't used by the credit-default experiment, and the main test suite doesn't run it.

- `appetite_guide.md`: one-page guide for an invented small-commercial property insurer ("Quillfield Mutual").
- `rules.py`: pure rule engine (decline > refer > accept). Labels must come from this file, never from an LLM.
- `test_rules.py`: tests for every rule, both sides of each threshold, the precedence order, and guide/code consistency.

Run the tests on their own with `uv run pytest parked/insurance`.

Not built yet: the email generator, the extractor that re-reads the facts from each email, and the splits.

Notes from the original M1 research (2026-11-22), for when this comes back:
- ContractNLI is CC BY 4.0 (github.com/stanfordnlp/contract-nli); pin the zip to gh-pages commit `eced6528dd3c1d14d73f9a87df8f7bdbc03126f9`.
- CUAD insurance clauses: `mteb/CUADInsuranceLegalBenchClassification` at revision `d1e3806b0211cb14120dfb3aa2d668ef1140fc21`, CC BY 4.0. It has 1,030 test items and 6 train items, so the train pool has to come from the remaining test items.
- System One question types are `noul`, `choice` and `score`. For `choice`, the API's `confidence` field is `(p_max - 1/K)/(1 - 1/K)`, not `p_max`.
