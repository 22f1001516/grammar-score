# Spoken Grammar Scoring

Audio based grammar score prediction experiments for the SHL grammar scoring task.

## Contents

- `grammar_scoring.ipynb`, `gs2.ipynb` to `gs4.ipynb`, and `gsf.ipynb`: experiment notebooks.
- `run_repeated_nested_blend_cv.py`: repeated nested cross validation for WavLM blend choices.
- `make_candidate_blend_submission.py`: fit WavLM candidates and create a fixed 0.50 / 0.25 / 0.25 submission.
- `outputs/`: saved predictions, CV reports, model artifacts, and plots.

## Local data

The notebooks and scripts expect competition files under `data/raw/` and cached features under `cache/`. Those directories, along with the Python virtual environment, are excluded from Git because they are large generated or local files. Place the competition data and caches beside the notebooks before running experiments.

The scripts use Python with NumPy, pandas, and scikit-learn. The notebooks list additional audio and machine learning dependencies in their setup cells.
