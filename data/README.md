# Data

This repository does not redistribute the source PDFs or derived chunk text used in the experiments.

## Sources used in the current prototype

| Domain | Source | Notes |
|---|---|---|
| medical | AAOS Clinical Practice Guideline: *Management of Hip Fractures in Older Adults* | Obtain from the original publisher and follow its terms of use. |
| legal / regulatory | CMS guidance related to surgical informed consent | Obtain from the original CMS source. |

## Local layout

Place source files you are permitted to use under:

```text
data/
├── medical_raw/
│   └── *.pdf
└── legal_raw/
    └── *.pdf
```

Then generate local derived segments with:

```bash
python src/prepare_medical_data.py
python src/prepare_legal_data.py
```

Generated files such as `medical_segments.csv`, `legal_segments.csv`, local databases, raw PDFs, and evaluation run outputs are intentionally git-ignored because they may contain source-document text.
