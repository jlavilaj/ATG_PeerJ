import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    average_precision_score, roc_auc_score, log_loss
)


CSV_PATH = r""# <-- change if needed


df_raw = pd.read_csv(CSV_PATH, low_memory=False)
print("Loaded:", df_raw.shape)
print("Begin feature building:", datetime.now())

# -------------------------
# (B) Build student-level features (same as yours)
# -------------------------
def to_float_safe(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return np.nan

def build_student_features_kau(
    df,
    id_col="STD_STUDENT_ID",
    year_col="STD_STUDY_PERIOD_YEAR",
    sem_col="STD_STUDY_PERIOD_SEMESTER",
    grade_col="STD_GRADE",
    credits_col="CRS_CREDIT_HOURS",
    gpa_col="STD_UNIVERSITY_GPA",
    pass_mark=50,
    gpa_threshold=4.0
):
    out_rows = []
    df = df.copy()

    for c in [grade_col, credits_col, gpa_col, year_col, sem_col]:
        if c in df.columns:
            df[c] = df[c].apply(to_float_safe)

    static_cols = [
        "STD_GENDER_CODE",
        "STD_AGE_BY_YEAR",
        "STD_SCHOOL_TYPE",
        "STD_HIGH_SCHOOL_GPA",
        "STD_ADMYEAR",
        "STD_MAJOR_CODE",
        "STD_PROGRAM",
    ]
    static_cols = [c for c in static_cols if c in df.columns]

    for sid, g in df.groupby(id_col):
        if year_col in g.columns and sem_col in g.columns and g[year_col].notna().any():
            sem_order = g[year_col].fillna(0) * 10 + g[sem_col].fillna(0)
        else:
            sem_order = pd.Series(np.arange(len(g)), index=g.index)

        g = g.assign(_sem_order=sem_order).sort_values("_sem_order")

        # Label from FINAL GPA (last record)
        final_gpa = g[gpa_col].dropna().iloc[-1] if g[gpa_col].notna().any() else np.nan
        if np.isnan(final_gpa):
            continue
        label = 1 if final_gpa > gpa_threshold else 0

        # History BEFORE last semester when possible
        last_sem = g["_sem_order"].max()
        g_hist = g[g["_sem_order"] < last_sem]
        if len(g_hist) == 0:
            g_hist = g

        total_courses = len(g_hist)
        n_semesters = g_hist["_sem_order"].nunique()

        grades = g_hist[grade_col].dropna().values if grade_col in g_hist.columns else np.array([])
        if len(grades) == 0:
            passed_ratio = failed_ratio = avg_grade = 0.0
        else:
            passed = (grades >= pass_mark).sum()
            failed = (grades < pass_mark).sum()
            passed_ratio = passed / len(grades)
            failed_ratio = failed / len(grades)
            avg_grade = float(np.mean(grades))

        credits = g_hist[credits_col].dropna() if credits_col in g_hist.columns else pd.Series([], dtype=float)
        credits_attempted = float(credits.sum()) if len(credits) else 0.0
        if credits_col in g_hist.columns and grade_col in g_hist.columns:
            credits_passed = float(g_hist.loc[g_hist[grade_col] >= pass_mark, credits_col].dropna().sum())
        else:
            credits_passed = 0.0

        static_vals = {}
        for c in static_cols:
            val = g[c].dropna().iloc[0] if g[c].notna().any() else np.nan
            if c in ["STD_AGE_BY_YEAR", "STD_HIGH_SCHOOL_GPA", "STD_ADMYEAR"]:
                val = to_float_safe(val)
            static_vals[c] = val

        out_rows.append({
            "STD_STUDENT_ID": sid,
            "n_courses_hist": total_courses,
            "n_semesters_hist": n_semesters,
            "passed_ratio_hist": passed_ratio,
            "failed_ratio_hist": failed_ratio,
            "avg_grade_hist": avg_grade,
            "credits_attempted_hist": credits_attempted,
            "credits_passed_hist": credits_passed,
            **static_vals,
            "label": label
        })

    return pd.DataFrame(out_rows)

data = build_student_features_kau(df_raw)
data = data.dropna(subset=["label"]).reset_index(drop=True)

y = data["label"].astype(int).values
X_df = data.drop(columns=["label", "STD_STUDENT_ID"])

categorical_cols = [c for c in X_df.columns if X_df[c].dtype == "object"]
numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

print("Student-level table:", data.shape)
print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)
print("Class balance -> Pos:", int(y.sum()), "Neg:", int((1-y).sum()))
print("End feature building:", datetime.now())

# -------------------------
# (C) Preprocessor (same idea)
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0,
)

# -------------------------
# (D) Traditional baselines
# - DummyClassifier: أضعف baseline (غالبًا أسوأ من GNN)
# - LogisticRegression / SVM / RandomForest: baselines تقليدية
# -------------------------
models = {
    "Dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
    "LogReg": LogisticRegression(max_iter=2000, n_jobs=None),
    "LinearSVM": SVC(kernel="linear", probability=True),  # probability for AUC/PR-AUC
    "RandomForest": RandomForestClassifier(
        n_estimators=500, random_state=42, n_jobs=-1
    ),
}

def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_prob)

def safe_pr_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_prob)

# -------------------------
# (E) CV protocol (same splits philosophy)
# -------------------------
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

all_results = {name: [] for name in models.keys()}

print("\nBegin Experiments:", datetime.now())

for fold, (train_idx, test_idx) in enumerate(skf.split(X_df, y), start=1):
    print(f"\n===== Fold {fold}/{n_folds} =====")

    X_train_all = X_df.iloc[train_idx].copy()
    y_train_all = y[train_idx]
    X_test_df   = X_df.iloc[test_idx].copy()
    y_test      = y[test_idx]

    # نفس فكرة val split (للتطابق فقط)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_sub_idx, val_sub_idx = next(sss.split(X_train_all, y_train_all))

    X_train_df = X_train_all.iloc[tr_sub_idx].copy()
    y_train    = y_train_all[tr_sub_idx]

    # Fit لكل موديل على train فقط
    for name, clf in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", clf),
        ])

        pipe.fit(X_train_df, y_train)

        # probabilities
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_prob = pipe.predict_proba(X_test_df)[:, 1]
        else:
            # fallback: decision_function -> sigmoid تقريبًا (لكن هنا غالبًا لن نحتاجه)
            scores = pipe.decision_function(X_test_df)
            y_prob = 1 / (1 + np.exp(-scores))

        y_pred = (y_prob >= 0.5).astype(int)

        res = {}
        # loss (log loss)
        res["loss"] = log_loss(y_test, y_prob, labels=[0,1]) if len(np.unique(y_test)) > 1 else 0.0
        res["acc"] = accuracy_score(y_test, y_pred)
        res["auc"] = safe_auc(y_test, y_prob)
        res["precision"] = precision_score(y_test, y_pred, zero_division=0)
        res["recall"] = recall_score(y_test, y_pred, zero_division=0)
        res["f1"] = f1_score(y_test, y_pred, zero_division=0)
        res["balanced_acc"] = balanced_accuracy_score(y_test, y_pred)
        res["mcc"] = matthews_corrcoef(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.0
        res["pr_auc"] = safe_pr_auc(y_test, y_prob)

        all_results[name].append(res)

        print(name, "->", {k: round(v, 4) for k, v in res.items()})

print("\nEnd Experiments:", datetime.now())

# -------------------------
# (F) Summary mean ± std
# -------------------------
print("\n===== CV Summary (mean ± std) =====")
for name, rows in all_results.items():
    df_res = pd.DataFrame(rows)
    print(f"\n### {name}")
    for col in ["loss","acc","auc","precision","recall","f1","balanced_acc","mcc","pr_auc"]:
        print(f"{col}: {df_res[col].mean():.4f} ± {df_res[col].std():.4f}")
