# Developer Salary Prediction

Estimates a software developer's annual salary from their country, role, education, work arrangement, and years of professional experience. It gives a typical (median) figure and the range that half of similar developers fall in. Trained on the 2022 Stack Overflow Developer Survey.

**Live demo:** https://snehalbhatnagar.github.io/SalaryPrediction/ (the model runs in the browser)

## Project history

I first built this project in 2022 as a Streamlit app backed by MySQL. The original code was lost before it was pushed to GitHub. This repository is a **2026 rebuild**, written with help from Claude (see commit history). It keeps the original idea and stack (Python, scikit-learn, Streamlit, MySQL) and adds held-out evaluation, prediction ranges, tests, and a browser version.

## Results

Evaluated on 5,277 responses held out from training:

| Metric | Value |
| --- | --- |
| Mean absolute error | $26,299 |
| Median absolute error | $14,826 |
| Baseline: predict each country's median salary, mean absolute error | $30,525 |
| R² on log salary | 0.64 |
| Actual salaries inside the predicted 25th–75th percentile band | 50% |

The model beats the country-median baseline by about $4,200 on average. The 25th–75th band is well calibrated: about half of real salaries land inside it, which is what a 50% band should do.

## How it works

1. **Clean** (`src/prepare_data.py`, `src/features.py`): keep full-time professional developers with annual pay from $5k to $400k USD. Group education into four levels. Pick one primary role from the multi-select answer, preferring the most specific (manager over data engineer over back-end, for example). Group countries with fewer than 200 responses as "Other". 73,268 responses become 26,382 rows.
2. **Train** (`src/train.py`): scikit-learn gradient-boosted trees (300 trees, depth 3) on log salary, plus two quantile-loss models for the 25th and 75th percentiles. 80/20 train/test split with a fixed seed.
3. **Export**: the fitted trees are written to `docs/model.json`. `docs/app.js` walks them in the browser; the training script and the test suite check that the exported trees reproduce scikit-learn's predictions.
4. **Serve**: `app.py` is a Streamlit app with Predict, Explore, History, and About tabs. Each prediction is saved through SQLAlchemy to SQLite by default, or to MySQL when `DATABASE_URL` is set.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py                      # http://localhost:8501

# Use MySQL for prediction history instead of SQLite
export DATABASE_URL="mysql+pymysql://user:password@localhost:3306/salary_prediction"
```

Retrain from scratch (see `data/README.md` for the raw download):

```bash
python src/prepare_data.py data/raw/survey_results_public.csv
python src/train.py                       # updates models/ and docs/
python -m pytest
```

## Limits

Pay is self-reported for 2022 and converted to USD by Stack Overflow. Survey respondents are not a random sample of developers. City, company, industry, and specific skills are not in the model, and they explain much of the remaining error.

## Data and license

Data: [Stack Overflow Developer Survey 2022](https://survey.stackoverflow.co/2022/), Open Database License (ODbL). See `data/README.md`.
