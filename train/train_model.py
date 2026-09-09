"""
VittaBodh — Model Training Script
Trains the Deep Neural Network described in the paper:
  Input(25) -> Dense(128, ReLU, Dropout 0.3) -> Dense(64, ReLU, Dropout 0.3)
  -> Dense(32, ReLU) -> Dense(1, Sigmoid)

Enhancements over the original:
  - ROC-AUC added to metrics.json
  - Platt-scaling calibrator trained on the validation fold and persisted
    to probability_calibrator.pkl so the Flask app can emit calibrated
    probabilities (instead of raw sigmoid outputs).
  - Feature statistics (mean + std per numeric feature) saved to
    feature_stats.json for the drift detector.
  - Training loss / val_loss history saved for offline analysis.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Loan_Prediction_dataset.xlsx")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CATEGORICAL_COLS = [
    "Gender", "MaritalStatus", "Education", "Nationality", "Region",
    "EmploymentType", "IndustrySector", "LoanPurpose", "GuarantorAvailable",
]
NUMERIC_COLS = [
    "Age", "YearsExperience", "MonthlyIncome", "MonthlyExpenses",
    "SavingsBalance", "InvestmentValue", "ExistingEMI", "CreditScore",
    "PastDefaults", "LatePayments", "ExistingLoans", "CurrentJobDuration",
    "Dependents", "LoanAmount", "LoanTenure", "CollateralValue",
]
FEATURE_ORDER = NUMERIC_COLS + CATEGORICAL_COLS   # 16 + 9 = 25
TARGET_COL = "LoanApproved"


def load_data():
    df = pd.read_excel(DATA_PATH)
    df = df.drop(columns=["CustomerID"], errors="ignore")
    return df


def build_preprocessors(df):
    encoders = {}
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    X = df_enc[FEATURE_ORDER].astype(float)
    y = df_enc[TARGET_COL].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, encoders, scaler, X


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu", name="hidden_1"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu", name="hidden_2"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu", name="hidden_3"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_platt_calibrator(raw_probs_val, y_val):
    """Fit a logistic regression on top of the DNN raw probabilities (Platt scaling)."""
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(raw_probs_val.reshape(-1, 1), y_val)
    return calibrator


def main():
    print("Loading dataset...")
    df = load_data()
    print(f"  {df.shape[0]} records, {df.shape[1] - 1} raw columns")

    X_scaled, y, encoders, scaler, X_raw = build_preprocessors(df)

    # 80% train+val, 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    # Split train+val into train / calibration-val (85/15 of the 80%)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.15, random_state=RANDOM_STATE, stratify=y_trainval
    )

    print("Building model...")
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
    )

    print("Training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )

    print("Evaluating on held-out test set...")
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "precision":        float(precision_score(y_test, y_pred)),
        "recall":           float(recall_score(y_test, y_pred)),
        "f1_score":         float(f1_score(y_test, y_pred)),
        "roc_auc":          float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_size":        int(len(y_test)),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

    # -- Platt scaling calibrator -------------------------------------------
    print("Training Platt calibrator on calibration split...")
    raw_cal_probs = model.predict(X_cal, verbose=0).ravel()
    calibrator = train_platt_calibrator(raw_cal_probs, y_cal)
    cal_probs_test = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    cal_pred_test = (cal_probs_test >= 0.5).astype(int)
    print(f"  Calibrated accuracy: {accuracy_score(y_test, cal_pred_test):.4f}")
    print(f"  Calibrated ROC-AUC:  {roc_auc_score(y_test, cal_probs_test):.4f}")

    # -- Feature statistics for drift detection -----------------------------
    print("Computing feature statistics for drift detection...")
    X_raw_numeric = X_raw[NUMERIC_COLS].astype(float)
    feature_stats = {}
    for col in NUMERIC_COLS:
        feature_stats[col] = {
            "mean": float(X_raw_numeric[col].mean()),
            "std":  float(X_raw_numeric[col].std()),
            "min":  float(X_raw_numeric[col].min()),
            "max":  float(X_raw_numeric[col].max()),
        }

    # -- Training history ---------------------------------------------------
    training_history = {
        "loss":     [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "accuracy": [float(v) for v in history.history.get("accuracy", [])],
    }

    # -- Persist all artifacts ----------------------------------------------
    model.save(os.path.join(MODEL_DIR, "vittabodh_dnn.keras"))
    joblib.dump(scaler,      os.path.join(MODEL_DIR, "feature_scaler.pkl"))
    joblib.dump(encoders,    os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(FEATURE_ORDER, os.path.join(MODEL_DIR, "feature_order.pkl"))
    joblib.dump(calibrator,  os.path.join(MODEL_DIR, "probability_calibrator.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(MODEL_DIR, "feature_stats.json"), "w") as f:
        json.dump(feature_stats, f, indent=2)
    with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\nAll artifacts saved to {MODEL_DIR}/")
    print("  vittabodh_dnn.keras")
    print("  feature_scaler.pkl")
    print("  label_encoders.pkl")
    print("  feature_order.pkl")
    print("  probability_calibrator.pkl  [NEW]")
    print("  metrics.json               (+ roc_auc)")
    print("  feature_stats.json         [NEW — enables drift detection]")
    print("  training_history.json      [NEW]")


if __name__ == "__main__":
    main()
