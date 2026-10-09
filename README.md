# Sleep Disorder Classification

A reproducible machine-learning project that classifies a record as
**Healthy**, **Insomnia**, or **Sleep Apnea** from sleep, lifestyle, and
physiological features.

The deployable artifact is one scikit-learn pipeline containing feature
engineering, missing-value handling, encoding, scaling, and classification.
Training and prediction therefore apply exactly the same transformations.

## Results

First, one stratified, group-disjoint fold is reserved as a holdout. Models are
then compared with stratified, group-aware five-fold cross-validation on the
remaining training rows.
Rows with identical engineered model features stay in the same fold, preventing
equivalent patient profiles from appearing in both training and validation.
Grouping ignores unused columns and treats normalized BMI labels and blood
pressure formatting consistently.
Selection was based on mean macro F1, which gives equal importance to all
three classes.

| Model | Macro F1 | Mean accuracy | Fold accuracy SD |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.846 | 0.876 | 0.025 |
| **Support Vector Machine** | **0.855** | **0.889** | **0.020** |
| Gradient Boosting | 0.855 | 0.889 | 0.026 |

Support Vector Machine achieved the highest unrounded mean macro F1 and was
fitted on the 299 training rows for the saved pipeline. On the 75 reserved
holdout rows, it achieved **0.985 macro F1** and **0.987 accuracy**. The holdout
was not used to choose the model. Full per-class scores and a confusion matrix
are in [`reports/holdout_evaluation.json`](reports/holdout_evaluation.json).
These are internal dataset estimates, not results from an external test set.

![Model comparison](reports/model_comparison.png)

## What Was Corrected

- The model now saves preprocessing and classification as one pipeline.
- Training and prediction use identical feature transformations.
- `Person ID` is excluded because it is an identifier, not a clinical feature.
- Blood pressure is parsed into systolic and diastolic values, and severe
  hypertension is checked before the broader stage-2 condition.
- Categorical features use an encoder that safely handles unseen values.
- Numeric and categorical missing values are imputed inside each validation
  fold, preventing preprocessing leakage.
- Equivalent engineered feature rows are assigned to the same validation group
  to prevent duplicate-profile leakage, even when raw formatting or unused
  columns differ.
- Model selection uses stratified, group-aware five-fold cross-validation
  within a training partition; a group-disjoint holdout evaluates the winner.

## Repository Structure

```text
sleep-disorder-classification/
├── data/
│   └── README.md
├── models/
│   └── sleep_disorder_pipeline.joblib
├── reports/
│   ├── model_comparison.png
│   └── holdout_evaluation.json
├── src/
│   ├── train.py
│   ├── predict.py
│   └── preprocessing.py
├── tests/
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   └── test_training.py
├── .github/workflows/ci.yml
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## Dataset

The project uses the
[Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset).
The local training file contains 374 records and 13 columns. Missing target
values are interpreted as `Healthy`.

The dataset is published separately under **CC0: Public Domain**. It is not
covered by this repository's MIT License. Download instructions are available
in [`data/README.md`](data/README.md); CSV files are intentionally excluded
from Git.

## Installation

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/Dochikhoa2006/Sleeping-Disorder-Analysis.git
cd Sleeping-Disorder-Analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Train

Place the dataset at
`data/Sleep_health_and_lifestyle_dataset.csv`, then run:

```bash
python -m src.train
```

This command compares the three candidate models, then regenerates:

- `models/sleep_disorder_pipeline.joblib`
- `reports/model_comparison.png`
- `reports/holdout_evaluation.json`

Optional paths can be supplied with `--data`, `--model-output`,
`--report-output`, and `--evaluation-output`.

## Predict

Run the interactive predictor:

```bash
python -m src.predict
```

For a repeatable non-interactive prediction, create `patient.json`:

```json
{
  "Gender": "Male",
  "Age": 31,
  "Occupation": "Software Engineer",
  "Sleep Duration": 7.2,
  "Quality of Sleep": 8,
  "Physical Activity Level": 60,
  "Stress Level": 4,
  "BMI Category": "Normal",
  "Blood Pressure": "120/80",
  "Heart Rate": 70,
  "Daily Steps": 8000
}
```

Then run:

```bash
python -m src.predict --input-json patient.json
```

For several patients, supply a JSON array of objects with the same fields.
Predictions are returned in input order, and invalid records are identified by
their position. To save machine-readable results:

```bash
python -m src.predict --input-json patients.json --output-json predictions.json
```

The output file contains an array of predicted class names in the same order
as the input. A single patient object also works with `--output-json` and
produces a one-element array.

### CSV batch prediction

For spreadsheet exports, save a CSV with the same feature names as column
headers and one patient per row. Extra columns, such as a local record ID, are
preserved in the output:

```bash
python -m src.predict --input-csv patients.csv --output-csv predictions.csv
```

The output adds a `Predicted Sleep Disorder` column. Numeric fields must contain
numbers; blood pressure uses `systolic/diastolic` format. The command reports
the CSV row number for invalid records and does not write an output file until
all rows have passed validation. The input and output paths must differ.

Only load `.joblib` files from sources you trust. Joblib artifacts can execute
code when loaded.

## Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same test command with Python 3.11 for every push and
pull request.

## Docker

Build the inference image:

```bash
docker build -t sleep-disorder-classifier .
```

Run it interactively:

```bash
docker run --rm -it sleep-disorder-classifier
```

Or provide a JSON file:

```bash
docker run --rm \
  -v "$PWD/patient.json:/tmp/patient.json:ro" \
  sleep-disorder-classifier \
  --input-json /tmp/patient.json
```

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- SciPy
- Matplotlib and seaborn
- joblib

Exact tested dependency versions are pinned in `requirements.txt`.

## Limitations

- The dataset is small and synthetic, so reported scores may not generalize to
  real clinical populations.
- The model has not been clinically validated.
- Predictions depend on self-reported and simplified health features.
- Fold standard deviations describe variation among five correlated
  validation folds; they are not population confidence intervals or
  guarantees for future data.
- The holdout is one split of a small synthetic dataset; its high score may
  vary substantially with a different split and needs confirmation on external
  data. Cross-validation scores used for selection may still be optimistic.

## License

The source code in this repository is licensed under the
[MIT License](LICENSE).

The dataset is a separate work distributed by its publisher under
[CC0: Public Domain](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset).
The MIT License does not apply to the dataset.

## Citation

Do, Chi Khoa (2026). *Sleep Health and Lifestyle Dataset Analysis and
Prediction*.

## Contact

Chi Khoa Do — dochikhoa2006@gmail.com
