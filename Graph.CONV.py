# GCN - Correct Evaluation Protocol (NO Data Leakage)
# - NEW model per fold (no cross-fold weight carryover)
# - Validation split from TRAIN only (no using test as validation)
# - StratifiedKFold (better for classification)
# - Optional: best threshold chosen on validation (no leakage)
# - Reports mean ± std (and extra imbalance-aware metrics)

import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (f1_score,balanced_accuracy_score,matthews_corrcoef,
    average_precision_score,precision_score,recall_score,)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import binary_crossentropy
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph.mapper import PaddedGraphGenerator
#import Aux_functions
#from Aux_functions import create_atg_graph

import importlib
import ATG
importlib.reload(ATG)
create_atg_graph = ATG.create_atg_graph



CSV_PATH = r""# <-- change if needed

kau_data = pd.read_csv(CSV_PATH)

print("Begin Graph creation:", datetime.now())

students_ids = kau_data["STD_STUDENT_ID"].unique()

#graphs = []
#student_labels = []
#threshold_GPA = 4.0  # Student is "good" if GPA > 4
#students_ids = kau_data["STD_STUDENT_ID"].unique()

graphs = []
student_labels = []
threshold_GPA = 4.0
for sid in students_ids:
    student_data = kau_data[kau_data["STD_STUDENT_ID"] == sid]
    g = create_atg_graph(student_data)
    graphs.append(g)



    # Create label from GPA
    gpa_str = str(student_data["STD_UNIVERSITY_GPA"].iloc[0]).replace(",", ".")
    gpa = float(gpa_str)
    student_labels.append(1 if gpa > threshold_GPA else 0)

graphs = np.array(graphs, dtype=object)
student_labels = np.array(student_labels)

print("Number of students:", len(student_labels))
print("End Graph creation:", datetime.now())

# (B) Generator (built on ALL graphs)
generator = PaddedGraphGenerator(graphs=graphs)

# (C) Model factory: NEW model per fold
def get_model(generator, lr=0.0005):
    layer_sizes = [32, 32, 32]

    gcn = GCNSupervisedGraphClassification(layer_sizes=layer_sizes,activations=["tanh", "tanh", "tanh"],
        generator=generator,dropout=0.2,)

    x_inp, x_out = gcn.in_out_tensors()
    x_out = Dense(units=1024, activation="relu")(x_out)
    x_out = Dropout(rate=0.2)(x_out)
    predictions = Dense(units=1, activation="sigmoid")(x_out)

    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=lr),loss=binary_crossentropy,metrics=["acc",
            keras.metrics.AUC(name="auc"),keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),],)
    return model


# (D) Threshold selection on validation (NO leakage)
def find_best_threshold(y_true, y_prob, metric="f1"):
    """
    Choose threshold that maximizes a metric on validation data.
    metric: "f1" or "balanced_acc"
    """
    thresholds = np.linspace(0.05, 0.95, 19)  # 0.05, 0.10, ..., 0.95
    best_t, best_score = 0.5, -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "balanced_acc":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


n_folds = 10
EPOCHS = 150
BATCH_SIZE = 50

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_results = []

print("Begin Experiments:", datetime.now())

for fold, (train_index, test_index) in enumerate(skf.split(graphs, student_labels), start=1):
    print(f"\n===== Fold {fold}/{n_folds} =====")

    X_train_all = graphs[train_index]
    y_train_all = student_labels[train_index]
    X_test = graphs[test_index]
    y_test = student_labels[test_index]

    # Show class distribution in test fold (useful for imbalance reporting)
    pos = int(y_test.sum())
    neg = int(len(y_test) - pos)
    print(f"Test class distribution -> good(1): {pos}, bad(0): {neg}")

    # Validation split FROM TRAIN only (never use test as validation)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_sub_idx, val_sub_idx = next(sss.split(X_train_all, y_train_all))

    X_train = X_train_all[tr_sub_idx]
    y_train = y_train_all[tr_sub_idx]
    X_val = X_train_all[val_sub_idx]
    y_val = y_train_all[val_sub_idx]

    train_gen = generator.flow(graphs=X_train, targets=y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_gen   = generator.flow(graphs=X_val,   targets=y_val,   batch_size=BATCH_SIZE, shuffle=False)
    test_gen  = generator.flow(graphs=X_test,  targets=y_test,  batch_size=BATCH_SIZE, shuffle=False)

    # IMPORTANT: NEW model per fold (prevents cross-fold leakage)
    model = get_model(generator, lr=0.0005)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    ]

    model.fit(
        train_gen,
        epochs=EPOCHS,
        verbose=1,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # ---- Choose best threshold on VALIDATION (NO leakage) ----
    y_val_prob = model.predict(val_gen, verbose=0).reshape(-1)
    best_t, best_val_score = find_best_threshold(y_val, y_val_prob, metric="f1")
    print(f"Best threshold (from val) = {best_t:.2f}, val_f1 = {best_val_score:.4f}")

    # ---- Evaluate on TEST ----
    # Keras evaluate gives acc/auc/precision/recall (typically using threshold=0.5 internally for precision/recall/acc)
    base_metrics = model.evaluate(test_gen, verbose=0)
    result = dict(zip(model.metrics_names, base_metrics))

    # Our thresholded metrics (using best_t)
    y_test_prob = model.predict(test_gen, verbose=0).reshape(-1)
    y_test_pred = (y_test_prob >= best_t).astype(int)

    result["best_t"] = best_t
    result["f1"] = f1_score(y_test, y_test_pred, zero_division=0)
    result["balanced_acc"] = balanced_accuracy_score(y_test, y_test_pred)
    result["mcc"] = matthews_corrcoef(y_test, y_test_pred) if len(np.unique(y_test)) > 1 else 0.0
    result["pr_auc"] = average_precision_score(y_test, y_test_prob) if len(np.unique(y_test)) > 1 else 0.0

    # (Optional) precision/recall at best_t (consistent with chosen threshold)
    result["precision_t"] = precision_score(y_test, y_test_pred, zero_division=0)
    result["recall_t"] = recall_score(y_test, y_test_pred, zero_division=0)

    print("Test metrics:", {k: round(v, 4) for k, v in result.items()})
    fold_results.append(result)

print("\nEnd Experiments:", datetime.now())


# =========================
# (F) Summary: mean ± std across folds
# =========================
df = pd.DataFrame(fold_results)

print("\n===== Cross-Validation Summary (mean ± std) =====")
for col in df.columns:
    print(f"{col}: {df[col].mean():.4f} ± {df[col].std():.4f}")