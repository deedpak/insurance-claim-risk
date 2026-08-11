# 🔍 Insurance Claim Risk Detection

**AI-powered fraud risk scoring for auto-insurance claims.**

**🚀 [Live App](https://insurance-claim-risk-ap7ujahjub93vknofkwmsh.streamlit.app/) &nbsp;·&nbsp; 📦 [This Repo](https://github.com/deedpak/insurance-claim-risk)**

---

## What it does

An insurer processes thousands of claims manually. This prototype flags the ones investigators should look at first — and explains *why* — in four steps:

| | |
|---|---|
| 🧹 **Clean & understand** | 1,000 real auto-insurance claims, 39 fields — missing values, duplicates, and data leakage handled and documented |
| 📊 **Explore & engineer** | 7 visual analyses answering the key fraud questions, plus 9 engineered features |
| 🤖 **Model & explain** | Baseline → Logistic Regression → Random Forest, compared honestly with imbalance-aware metrics, explained per-claim with SHAP |
| 💬 **Summarise with AI** | Google Gemini turns each claim's model output into a plain-English note for a human investigator — it never decides fraud itself |

Everything is wrapped in an interactive **Streamlit** app: upload a CSV, see data quality and model performance, filter to high-risk claims, drill into any individual claim's SHAP breakdown and AI summary, and download the scored results.

## Screenshots

| Data quality & scoring | Individual claim explanation |
|---|---|
| *Upload a CSV to see missing values, duplicates, and live fraud scoring* | *SHAP factors + AI-generated investigator summary per claim* |

*(Add your own screenshots here — drag them into this README on GitHub, or reference an `assets/` folder.)*

## Live demo

No setup needed — try it directly:
**[insurance-claim-risk-ap7ujahjub93vknofkwmsh.streamlit.app](https://insurance-claim-risk-ap7ujahjub93vknofkwmsh.streamlit.app/)**

Upload `data/insurance_claims.csv` (included in this repo) to see it in action.

## Project structure

```
insurance-claim-risk/
├── data/
│   ├── insurance_claims.csv              # original dataset (1,000 claims, 39 fields)
│   ├── insurance_claims_cleaned.csv      # after notebook 1
│   └── insurance_claims_features.csv     # after notebook 3
├── notebooks/
│   ├── 01_data_understanding_cleaning.ipynb   # column dictionary, missing values, leakage
│   ├── 02_eda.ipynb                           # 7 visual analyses + interpretations
│   ├── 03_feature_engineering.ipynb           # 9 engineered features, justified
│   └── 04_modeling_explainability.ipynb       # models, metrics, SHAP, saves the final model
├── models/
│   ├── fraud_model.pkl                   # trained Random Forest
│   ├── feature_columns.pkl               # exact column order the model expects
│   └── categorical_columns.pkl
├── app/
│   └── app.py                            # Streamlit app (scoring, SHAP, AI summaries)
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick start

### 1. Clone and set up a virtual environment
```bash
git clone https://github.com/deedpak/insurance-claim-risk.git
cd insurance-claim-risk
python -m venv venv

venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Regenerate the pipeline from scratch
The trained model is already included in `models/`, so this step is optional. To rebuild it yourself, open each notebook in order and run all cells:
```
01_data_understanding_cleaning.ipynb   →  02_eda.ipynb   →  03_feature_engineering.ipynb   →  04_modeling_explainability.ipynb
```

### 4. Set your Gemini API key (for live AI explanations)
Free key: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

```bash
setx GEMINI_API_KEY "your-key-here"          # Windows, persists across sessions
export GEMINI_API_KEY="your-key-here"        # macOS / Linux
```
Without a key, the app still works fully — it falls back to a clear rule-based summary instead of a live AI-generated one.

### 5. Run the app
```bash
streamlit run app/app.py
```
Open the printed URL (usually `http://localhost:8501`) and upload `data/insurance_claims.csv`.

## Model performance

Final model: **Random Forest** (`class_weight='balanced'`, 5-fold cross-validated).

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Baseline (majority class) | — | 0.00 | — | 0.50 |
| Logistic Regression | see notebook 4 | | | |
| **Random Forest (final)** | see notebook 4 | | | |

*(Exact figures are in `notebooks/04_modeling_explainability.ipynb`, computed on a held-out 20% test set the model never saw during training.)*

**Why not just use accuracy?** Roughly 1 in 4 claims in this dataset is fraudulent. A model that always predicts "genuine" would already look ~75% accurate while catching zero fraud — so evaluation here is based on precision, recall, F1, and ROC-AUC instead.

## Key design decisions

- **Data leakage guarded against**: `policy_number`, `insured_zip`, and `incident_location` are dropped before modelling — unique identifiers that would let the model memorise records instead of learning generalisable patterns.
- **Class imbalance**: handled via `class_weight='balanced'` rather than naive oversampling, keeping the training distribution honest.
- **Schema differences from the assignment brief**: this real dataset doesn't include a few fields the brief's *example* schema lists (a separate claim-submission date, a previous-claim-count field, separate repair-estimate/invoice fields). Where this happened, the closest available proxy was used and documented inline in the relevant notebook (e.g. policy tenure at incident time as a stand-in for submission delay).
- **AI explanation boundaries**: the LLM receives only the model's numeric output — risk probability, category, and named SHAP factors — never raw claim text. It's explicitly prompted not to make a fraud determination, only to explain the model's signals, and the app labels every AI output as "not a decision" in the UI.

## Tech stack

**Python** · **pandas** / **NumPy** — data wrangling
**scikit-learn** — modelling (Logistic Regression, Random Forest)
**SHAP** — per-prediction explainability
**Streamlit** — interactive web app
**Google Gemini API** — generative AI investigator summaries

## Possible next steps

- Hyperparameter tuning (GridSearch/Optuna) on the Random Forest
- XGBoost as a fourth model for comparison
- Batch AI summaries (one Gemini call per upload instead of per claim view) to reduce latency
- User authentication for the Streamlit app before any real deployment

---

Built by **Deepak**.
