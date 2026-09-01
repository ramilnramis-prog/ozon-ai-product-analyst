# Ozon AI Product Analyst

Portfolio project #3 — AI-assisted marketplace product analytics system.

## Overview

Ozon AI Product Analyst analyzes marketplace XLSX/CSV exports and identifies promising product opportunities.

A user uploads marketplace data through a Streamlit interface. The system processes the file, classifies products into meaningful competitive niches, calculates deterministic business metrics, and displays a global TOP plus TOPs by marketplace subcategory.

## Core principle

Python calculates business facts and scores. AI is used for semantic tasks only.

```text
Python calculates.
AI interprets.
```

AI does not calculate the final business opportunity score.

## Pipeline

```text
XLSX / CSV
→ ingestion
→ semantic column mapping
→ validation
→ normalization and cleaning
→ product/category attributes
→ functional-family classification
→ niche grouping
→ competition analytics
→ candidate features
→ deterministic scoring and eligibility
→ ranking
→ grounded AI interpretation
→ Streamlit UI
```

## Automatic category classification

The project does not require a manually hard-coded taxonomy of all marketplace categories.

```text
known category
→ cached rules
→ deterministic Python classification

unknown category
→ real product examples
→ AI semantic analysis
→ structured functional-family rules
→ Python quality checks
→ runtime cache
→ deterministic Python classification thereafter
```

If a known category later contains a genuinely missing product family, the cache can be enriched. New rules are accepted only when deterministic quality checks confirm that classification quality does not regress.

## Functional families and niches

A broad marketplace category is not automatically treated as one competitive niche.

```text
category
→ functional_family
→ product_role where applicable
→ niche_key
→ competition analytics
```

## Competition analytics

The system evaluates active sellers, strong sellers, market depth, concentration, seller shares, competition warnings, and niche-level demand/opportunity. SKU count is not treated as seller count. Weak niches may legitimately have no suitable TOP candidate.

## Streamlit interface

The UI supports XLSX/CSV upload, multiple files, automatic analysis, a global TOP, TOP by root and leaf category, eligible-only category TOPs, cached category reuse, and limited AI classification per UI run.

Run:

```powershell
python -m streamlit run streamlit_app.py
```

## Installation

Recommended: Python 3.13

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` and configure the OpenAI API key locally. Do not commit `.env`.

## Tests

```powershell
python -m pytest -q
```

Current verified checkpoint: `129 passed`.

## Important modules

```text
app/data_loader.py
app/column_mapping.py
app/normalizer.py
app/data_cleaner.py
app/product_attributes.py
app/category_classifier.py
app/category_classification_ai.py
app/niche_grouping.py
app/competition.py
app/candidate_features.py
app/scoring.py
app/ranking.py
app/reporting.py
app/multi_period.py
streamlit_app.py
```

## Runtime category cache

Learned category classifications are stored in `data/cache/category_classifications.json`. Known categories can therefore be reused without repeating the same AI classification work.

## MVP limitations

This is a portfolio MVP rather than a production SaaS. Current limitations include Streamlit instead of a separate frontend/backend, file-based runtime cache, first-time AI calls for unknown categories, external API latency, and no authentication/billing/multi-user concurrency.

## Project goal

A TOP candidate means a promising marketplace opportunity worth deeper investigation, not a guaranteed recommendation to launch a product.

The project demonstrates a hybrid AI + deterministic analytics architecture where AI handles changing marketplace semantics while Python remains responsible for reproducible business calculations.
