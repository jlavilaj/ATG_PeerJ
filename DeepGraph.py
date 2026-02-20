# DeepGraphCNN (DGCNN) with StellarGraph
# ============================================================

import numpy as np
import pandas as pd
import keras
import tensorflow as tf

from datetime import datetime
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy

from stellargraph.layer import DeepGraphCNN
from stellargraph.mapper import PaddedGraphGenerator

from ATG import create_atg_graph
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef, average_precision_score

# =========================
# (A) Model factory (NEW per fold)
# =========================
def build_model(generator, k=35, layer_sizes=[32, 32, 32, 1], lr=0.001):
    dgcnn_model = DeepGraphCNN(layer_sizes=layer_sizes,activations=["tanh", "tanh", "relu", "relu"],
        k=k, bias=False, kernel_initializer="normal",generator=generator,)

    x_inp, x_out = dgcnn_model.in_out_tensors()

    # IMPORTANT: kernel_size must not exceed k (fixes the reviewer-noted issue)
    conv1_kernel = min(sum(layer_sizes), k)

    x_out = Conv1D(filters=16, kernel_size=conv1_kernel, strides=conv1_kernel)(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)
    x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

    x_out = Flatten()(x_out)
    x_out = Dense(units=1024, activation="relu")(x_out)
    x_out = Dropout(rate=0.2)(x_out)
    predictions = Dense(units=1, activation="sigmoid")(x_out)

    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=binary_crossentropy,
        metrics=["acc",keras.metrics.AUC(name="auc"),keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),],)
    return model


# =========================
# (B) Load data + build graphs/labels
# =========================

CSV_PATH = r""# <-- change if needed


kau_data = pd.read_csv(CSV_PATH)

print("Begin Graph creation:", datetime.now())

students_ids = kau_data["STD_STUDENT_ID"].unique()

graphs = []
student_labels = []

threshold_GPA = 4.0  # success if GPA > 4

for sid in students_ids:
    student_data = kau_data[kau_data["STD_STUDENT_ID"] == sid]

    # Build one graph per student (your ATG)
    g = create_atg_graph(student_data)
    graphs.append(g)

    gpa_str = student_data["STD_UNIVERSITY_GPA"].unique()[0]
    gpa = float(str(gpa_str).replace(",", "."))  # safety if commas exist
    student_labels.append(1 if gpa > threshold_GPA else 0)

graphs = np.array(graphs, dtype=object)
student_labels = np.array(student_labels)

print("Number of students:", len(student_labels))
print("End Graph creation:", datetime.now())


# =========================
# - NEW model per fold
# - validation from train only
# - mean ± std across folds
# =========================
n_folds = 10
EPOCHS = 150
BATCH_SIZE = 50

# Generator built on ALL graphs to ensure consistent padding shapes
generator = PaddedGraphGenerator(graphs=graphs)

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

fold_results = []

print("Begin Experiments:", datetime.now())

for fold, (train_index, test_index) in enumerate(skf.split(graphs, student_labels), start=1):
    print(f"\n===== Fold {fold}/{n_folds} =====")

    X_train_all = graphs[train_index]
    y_train_all = student_labels[train_index]
    X_test = graphs[test_index]
    y_test = student_labels[test_index]

    # Validation split from TRAIN only (never use test as validation)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_sub_idx, val_sub_idx = next(sss.split(X_train_all, y_train_all))

    X_train = X_train_all[tr_sub_idx]
    y_train = y_train_all[tr_sub_idx]
    X_val = X_train_all[val_sub_idx]
    y_val = y_train_all[val_sub_idx]

    train_gen = generator.flow(graphs=X_train, targets=y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = generator.flow(graphs=X_val, targets=y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_gen = generator.flow(graphs=X_test, targets=y_test, batch_size=BATCH_SIZE, shuffle=False)

    # VERY IMPORTANT: re-initialize model inside EACH fold
    model = build_model(generator=generator, k=35, layer_sizes=[32, 32, 32, 1], lr=0.001)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)]

    model.fit(train_gen,epochs=EPOCHS,verbose=1,validation_data=val_gen,callbacks=callbacks,)
    # Evaluate built-in metrics
    metrics = model.evaluate(test_gen, verbose=0)
    result = dict(zip(model.metrics_names, metrics))

    # ---- Extra metrics for imbalanced data (computed from predictions) ----
    # Get predicted probabilities on the TEST fold
    y_prob = model.predict(test_gen, verbose=0).reshape(-1)

    # Convert probabilities to class labels using threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    # True labels for this fold (already available)
    y_true = y_test

    # Compute extra metrics
    result["f1"] = f1_score(y_true, y_pred, zero_division=0)
    result["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    result["mcc"] = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    result["pr_auc"] = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    print("Test metrics:", {k: round(v, 4) for k, v in result.items()})
    fold_results.append(result)









#metrics = model.evaluate(test_gen, verbose=0)
#names = model.metrics_names
#result = dict(zip(names, metrics))
#print("Test metrics:", {k: round(v, 4) for k, v in result.items()})
#fold_results.append(result)

#print("\nEnd Experiments:", datetime.now())

# Summary: mean ± std
df = pd.DataFrame(fold_results)
print("\n===== Cross-Validation Summary (mean ± std) =====")
for col in df.columns:
    print(f"{col}: {df[col].mean():.4f} ± {df[col].std():.4f}")
