### Repository snapshot

- Layout: `Dataset/`, `notebooks/`, `src/`.
- Canonical dataset: `Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv` (notebooks load this file directly).

### Big picture

- This is a single-repo ML project organized as notebooks for exploration and `src/` for reusable code. Notebooks (`notebooks/*.ipynb`) drive most workflows (EDA, preprocessing, evaluation) and sometimes define variables used by `src` code.
- `src/preprocessing.py` contains sklearn Pipeline builders and a `save_preprocessor()` that writes `model/preprocessor.pkl`.
- Expect data flow: CSV in `Dataset/` -> notebooks import/process -> optional use of `src` preprocessing utilities -> trained models / artifacts placed in `model/` (directory may be created at runtime).

### Key files to inspect

- `src/preprocessing.py` — numeric and categorical pipelines, `make_preprocessor()`, `get_preprocessed_data()` and `save_preprocessor()`.
- `notebooks/eda.ipynb` — defines `numeric_columns` (example: `["MonthlyCharges","TotalCharges","tenure"]`) and other column lists used by `src` code.
- `notebooks/data_preprocessing.ipynb` and `notebooks/model_evaluation.ipynb` — canonical workflow notebooks to run and adapt.

### Patterns and gotchas for AI agents

- Variable ownership: column lists like `numeric_columns` and `categorical_columns` are declared in notebooks, not in `src/preprocessing.py`. When editing or refactoring `src`, either accept these as function parameters or add module-level definitions — preserve compatibility with existing notebooks.
- File paths: notebooks currently use absolute paths to the CSV; prefer converting to workspace-relative paths when modifying notebooks (example path currently used in `tempCodeRunnerFile.py`).
- Artifact path: `save_preprocessor()` writes to `model/preprocessor.pkl`. Ensure any changes keep or migrate this artifact path, and create the `model/` directory when running scripts.
- Minimal dependency surface: code imports `pandas`, `numpy`, `scikit-learn` (Pipeline, ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder) and `joblib`. Use these packages for compatibility.

### Developer workflows (how to run things)

- Run notebooks interactively with Jupyter / VS Code Notebook. Open `notebooks/data_preprocessing.ipynb` then run cells in order.
- To run preprocessing code from `src` in a script or REPL, ensure `numeric_columns` and `categorical_columns` are defined or passed in. Example quick script:

```python
from src import preprocessing
# define column lists (copy from eda.ipynb)
numeric_columns = ["MonthlyCharges","TotalCharges","tenure"]
categorical_columns = [col for col in df.columns if col not in numeric_columns + ["customerID","Churn"]]
pre = preprocessing.make_preprocessor()
# then fit/transform as notebooks do
```

- Recommended install (to reproduce environment):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn joblib jupyter
```

### Editing guidelines for agents

- Small, focused changes: keep `src/` functions stable (signatures) to avoid breaking notebooks that import them. If you change a signature, update notebooks accordingly.
- Prefer adding optional parameters (e.g., `make_preprocessor(numeric_columns=None, categorical_columns=None)`) rather than removing notebook-defined globals.
- When adding files (e.g., `model/`), ensure relative paths are used so notebooks remain portable across platforms.

### What NOT to change without human confirmation

- Replace absolute dataset paths in notebooks only after confirming desired workspace behavior.
- Renaming `save_preprocessor()` target path or altering artifact format (joblib → pickle) without explicit migration steps.

### Quick reference examples

- Column example (from `notebooks/eda.ipynb`): `numeric_columns=["MonthlyCharges","TotalCharges","tenure"]`.
- Preprocessor save path: `model/preprocessor.pkl` (see `src/preprocessing.py`).

If any section is unclear or you want the file expanded with examples for running notebooks, CI suggestions, or an automated script to run preprocessing end-to-end, tell me which part to expand.
