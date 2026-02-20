from datetime import datetime
import numpy as np
import pandas as pd
import keras
import tensorflow as tf

from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (f1_score,balanced_accuracy_score, matthews_corrcoef,average_precision_score,
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from stellargraph.data import BiasedRandomWalk
from ATG import create_atg_graph


# (A) Random-walk utilities
def generate_walks(graphs_list, walk_length=10, num_walks=2, p=0.5, q=2.0):
    """
    Generate random walks from a list of graphs.
    Returns: list of walks (each walk is list[str])
    """
    all_walks = []
    for g in graphs_list:
        walker = BiasedRandomWalk(g)
        walks = walker.run(nodes=list(g.nodes()),length=walk_length,n=num_walks,p=p,q=q,)
        walks = [[str(node) for node in w] for w in walks]
        all_walks.extend(walks)
    return all_walks


def train_word2vec_on_walks(walks, dimensions=32, window_size=5, epochs=3, workers=1):
    """
    Train Word2Vec ONLY on training walks (no leakage).
    """
    w2v = Word2Vec(sentences=walks,vector_size=dimensions,window=window_size,min_count=0,
        sg=1,workers=workers,epochs=epochs,)
    return w2v



def graph_embedding_from_w2v(graph, w2v_model):
    """
    Graph embedding = mean of node embeddings.
    If node not in vocab (rare), ignore it.
    """
    vecs = []
    for node in graph.nodes():
        key = str(node)
        if key in w2v_model.wv:
            vecs.append(w2v_model.wv[key])
    if len(vecs) == 0:
        # fallback (shouldn't happen normally)
        return np.zeros(w2v_model.vector_size, dtype=float)
    return np.mean(vecs, axis=0)


# =========================
# (B) MLP model factory (NEW per fold)
# =========================
def create_mlp(input_dim, lr=0.001):
    model = Sequential([
        Dense(1024, activation="relu", input_dim=input_dim),
        Dropout(0.3),
        Dense(1024, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["acc",keras.metrics.AUC(name="auc"),keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),],)
    return model


# =========================
# (C) Threshold selection (from validation only)
# =========================
def find_best_threshold(y_true, y_prob, metric="f1"):
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


# =========================
# (D) Load data + build graphs/labels ONCE
# =========================

CSV_PATH = r""






kau_data = pd.read_csv(CSV_PATH)

print("Begin Graph creation:", datetime.now())

students_ids = kau_data["STD_STUDENT_ID"].unique()

graphs = []
student_labels = []
threshold_GPA = 4.0

for sid in students_ids:
    student_data = kau_data[kau_data["STD_STUDENT_ID"] == sid]
    g = create_atg_graph(student_data)
    graphs.append(g)

    gpa_str = student_data["STD_UNIVERSITY_GPA"].unique()[0]
    gpa = float(str(gpa_str).replace(",", "."))
    student_labels.append(1 if gpa > threshold_GPA else 0)

graphs = np.array(graphs, dtype=object)
student_labels = np.array(student_labels)

print("Number of students:", len(student_labels))
print("End Graph creation:", datetime.now())


# =========================
# (E) Correct CV 
# =========================
n_folds = 10
EPOCHS = 50

# random-walk hyperparameters (keep same as your original unless you want to tune)
DIM = 128
WALK_LEN = 8
NUM_WALKS = 5
WINDOW = 10
P = 0.5
Q = 2.0

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_results = []

print("Begin Experiments:", datetime.now())

for fold, (train_index, test_index) in enumerate(skf.split(graphs, student_labels), start=1):
    print(f"\n===== Fold {fold}/{n_folds} =====")

    X_train_all = graphs[train_index]
    y_train_all = student_labels[train_index]
    X_test = graphs[test_index]
    y_test = student_labels[test_index]

    # validation split FROM TRAIN only
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_sub_idx, val_sub_idx = next(sss.split(X_train_all, y_train_all))

    X_train = X_train_all[tr_sub_idx]
    y_train = y_train_all[tr_sub_idx]
    X_val = X_train_all[val_sub_idx]
    y_val = y_train_all[val_sub_idx]

    # ---- Train Word2Vec ONLY on TRAIN graphs (NO leakage) ----
    train_walks = generate_walks(X_train, walk_length=WALK_LEN, num_walks=NUM_WALKS, p=P, q=Q)
    w2v = train_word2vec_on_walks(train_walks, dimensions=DIM, window_size=WINDOW, epochs=10, workers=2)

    # ---- Build embeddings using this fold’s W2V ----
    X_train_emb = np.vstack([graph_embedding_from_w2v(g, w2v) for g in X_train])
    X_val_emb   = np.vstack([graph_embedding_from_w2v(g, w2v) for g in X_val])
    X_test_emb  = np.vstack([graph_embedding_from_w2v(g, w2v) for g in X_test])

    # ---- NEW MLP model per fold (NO leakage) ----
    model = create_mlp(input_dim=X_train_emb.shape[1], lr=0.001)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    ]

    model.fit(
        X_train_emb, y_train,
        validation_data=(X_val_emb, y_val),
        epochs=EPOCHS,
        verbose=1,
        shuffle=True,
        callbacks=callbacks,
    )

    # ---- Choose best threshold on VALIDATION (NO leakage) ----
    y_val_prob = model.predict(X_val_emb, verbose=0).reshape(-1)
    best_t, best_val_score = find_best_threshold(y_val, y_val_prob, metric="f1")
    print(f"Best threshold (from val) = {best_t:.2f}, val_f1 = {best_val_score:.4f}")

    # ---- Evaluate on TEST ----
    base_metrics = model.evaluate(X_test_emb, y_test, verbose=0)
    result = dict(zip(model.metrics_names, base_metrics))  # loss, acc, auc, precision, recall

    y_test_prob = model.predict(X_test_emb, verbose=0).reshape(-1)
    y_test_pred = (y_test_prob >= best_t).astype(int)

    # Extra imbalance-aware metrics (with best_t)
    result["f1"] = f1_score(y_test, y_test_pred, zero_division=0)
    result["balanced_acc"] = balanced_accuracy_score(y_test, y_test_pred)
    result["mcc"] = matthews_corrcoef(y_test, y_test_pred) if len(np.unique(y_test)) > 1 else 0.0
    result["pr_auc"] = average_precision_score(y_test, y_test_prob) if len(np.unique(y_test)) > 1 else 0.0

    print("Test metrics:", {k: round(v, 4) for k, v in result.items()})
    fold_results.append(result)

print("\nEnd Experiments:", datetime.now())


# =========================
# (F) Summary: mean ± std (same metrics as you requested)
# =========================
df = pd.DataFrame(fold_results)

wanted_cols = ["loss", "acc", "auc", "precision", "recall", "f1", "balanced_acc", "mcc", "pr_auc"]
df = df[wanted_cols]

print("\n===== Cross-Validation Summary (mean ± std) =====")
for col in wanted_cols:
    print(f"{col}: {df[col].mean():.4f} ± {df[col].std():.4f}")