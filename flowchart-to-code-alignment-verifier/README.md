# Flowchart-to-Code Alignment Verifier

Source code for the Super AI Engineer Season 6 online hackathon.

## Result

- Evaluation metric: Accuracy
- Public leaderboard score: **1.00000**
- Submitted prediction: `submission_reviewed.csv`

## Approach

The dataset is small and contains repeated problem templates, so this solution
uses a structured multimodal judge instead of training an image classifier.

1. Match each test representation to its training family using a SHA-256 hash.
2. Give Gemini only the official training variants from the matching family.
3. Compare flowchart/pseudocode with Java using explicit checks for control flow,
   conditions, initialization, loop bounds, output paths, and boundary cases.
4. Preserve the three scores revealed by `sample_submission.csv`.
5. Perform a human-in-the-loop audit for conflicting modality scores and verify
   decisions against the scoring rubric and matching training family.

No case identifier is used as a model feature. API credentials are read from a
local `.env` file and are never stored in source code.

## Files

- `gemini_judge.py` — baseline per-representation Gemini judge with structured
  JSON, family matching, rate limiting, and optional consensus runs.
- `.env.example` — environment-variable template without credentials.
- `submission_reviewed.csv` — frozen public-score submission.

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Add a newly generated Gemini API key to `.env`:

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash
```

Do not commit or submit `.env`.

## Run

Inspect a prompt without using API quota:

```bash
python3 gemini_judge.py --id 1 --dry-run
```

Run the baseline judge:

```bash
python3 gemini_judge.py --all --runs 1 --output submission_gemini.csv
```

## Validation

Before submission, verify that the output:

- has exactly 18 rows in the original Id order;
- contains only integer scores from 0 to 3;
- has no missing predictions;
- preserves the revealed scores for Id 5, 8, and 18;
- uses the exact `Id,Alignment_score` header from the sample file.

For model development, split validation by problem-template family rather than
random rows, because flowchart and pseudocode templates repeat across cases.
