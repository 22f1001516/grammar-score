# Spoken Grammar Scoring

Audio based grammar score prediction experiments for the SHL grammar scoring task.

## Contents

- `grammar_scoring.ipynb`, `gs2.ipynb` to `gs4.ipynb`, and `gsf.ipynb`: experiment notebooks.
- `run_repeated_nested_blend_cv.py`: repeated nested cross validation for WavLM blend choices.
- `make_candidate_blend_submission.py`: fit the selected WavLM candidates and create a fixed 0.50 / 0.25 / 0.25 submission.
- `outputs/`: saved predictions, CV reports, model artifacts, and plots.

## Local data

The notebooks and scripts expect the competition files under `data/raw/` and cached features under `cache/`. Those folders are excluded from Git because they contain large audio data and generated feature files. The Python environment is also local and excluded. Create the folders and place the competition data and caches beside the notebooks before running experiments.

The scripts use Python with NumPy, pandas, and scikit-learn. The notebooks have additional audio and machine learning dependencies listed in their setup cells.
