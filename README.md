# Insurance Claim Risk Detection

**🚀 [Live App](https://insurance-claim-risk-ap7ujahjub93vknofkwmsh.streamlit.app/)** — try it directly in your browser, no setup needed.

A prototype that analyses insurance claims, predicts fraud risk, explains why each claim was flagged, and generates a plain-English investigator summary using AI — built as a 48-hour technical assessment.

## What this does

1. Cleans and analyses a real auto-insurance claims dataset (1000 claims, 39 fields)
2. Engineers features that strengthen fraud signal (policy age, claim ratios, missing-document count, etc.)
3. Trains and compares three models (baseline, Logistic Regression, Random Forest) with proper imbalanced-data evaluation
4. Explains every prediction using SHAP (which factors drove each claim's score, and by how much)
5. Uses Google's Gemini API to turn the model output into a short, plain-English summary for a human investigator
6. Presents everything through an interactive Streamlit web app

## Project structure

```
insurance-claim-risk/
├── data/
│   ├── insurance_claims.csv              # original dataset
│   ├── insurance_claims_cleaned.csv      # after notebook 1
│   └── insurance_claims_features.csv     # after notebook 3
├── notebooks/
│   ├── 01_data_understanding_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling_explainability.ipynb
├── models/
│   ├── fraud_model.pkl                   # trained Random Forest
│   ├── feature_columns.pkl               # exact column order the model expects
│   └── categorical_columns.pkl
├── app/
│   └── app.py                            # Streamlit app
├── requirements.txt
└── README.md
```

## Setup instructions

### 1. Clone this repo and create a virtual environment
```
git clone <your-repo-url>
cd insurance-claim-risk
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. (Optional but recommended) Run the notebooks in order
This regenerates the cleaned data, engineered features, and trained model from scratch:
```
notebooks/01_data_understanding_cleaning.ipynb
notebooks/02_eda.ipynb
notebooks/03_feature_engineering.ipynb
notebooks/04_modeling_explainability.ipynb
```
Open each in VS Code (or Jupyter) and run all cells in order. The `models/` folder is already included with a pre-trained model, so this step is optional if you just want to try the app.

### 4. Set your Gemini API key (for the AI explanation feature)
Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then:
```
$env:GEMINI_API_KEY="your-key-here"    # Windows PowerShell
# export GEMINI_API_KEY="your-key-here"  # Mac/Linux
```
If this isn't set, the app still works fully — it falls back to a rule-based summary instead of a live AI-generated one.

### 5. Run the app
```
streamlit run app/app.py
```
Open the URL it prints (usually `http://localhost:8501`), then upload `data/insurance_claims.csv` to try it out.

## Key design decisions & assumptions

- **Data leakage**: `policy_number`, `insured_zip`, and `incident_location` were dropped before modelling — they're unique identifiers that would let the model memorise records instead of learning generalisable patterns.
- **Class imbalance**: handled via `class_weight='balanced'` in both models, and evaluation is based on precision/recall/F1/ROC-AUC rather than accuracy alone (a model that always predicts "not fraud" would still look ~75% accurate while catching zero fraud).
- **Schema differences from the assignment brief**: the real dataset used here doesn't include a few fields the assignment's example schema lists (a separate "claim submission date", a "previous claim count" field, and separate repair-estimate/invoice fields). Where this happened, the closest available proxy was used and documented inline in the relevant notebook cell (e.g. policy tenure as a stand-in for claim-submission delay).
- **Final model**: Random Forest, chosen over Logistic Regression for its higher ROC-AUC and better handling of non-linear, mixed categorical/numeric relationships in the data. See notebook 4 for the full comparison table.
- **AI explanation boundaries**: the LLM only receives the model's numeric output (risk probability, category, SHAP factor names/values) — never raw claim text — and is explicitly prompted not to make a fraud determination, only to explain the model's signals. The app labels this clearly as "not a decision" in the UI.

## Model performance

See `notebooks/04_modeling_explainability.ipynb` for full metrics on the held-out test set (precision, recall, F1, ROC-AUC, confusion matrix, ROC/PR curves) and the business trade-off discussion between missed fraud vs. over-investigating genuine claims.

## Tech stack

Python, pandas, scikit-learn, SHAP, Streamlit, Google Gemini API (generative AI explanations).
