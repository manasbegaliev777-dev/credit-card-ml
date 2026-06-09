import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = "UCI_Credit_Card.csv"
MODELS_DIR = "models"


def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    df = df.drop(columns=["ID"])
    X = df.drop(columns=["default.payment.next.month"])
    y = df["default.payment.next.month"]
    return X, y


def evaluate(name: str, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"{name}: F1={f1:.4f}  ROC-AUC={auc:.4f}")


def main():
    X, y = load_and_preprocess(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- RandomForest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate("RandomForest (model_v1)", rf, X_test, y_test)
    joblib.dump(rf, f"{MODELS_DIR}/model_v1.pkl")
    print(f"Saved: {MODELS_DIR}/model_v1.pkl")

    # --- LogisticRegression ---
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    evaluate("LogisticRegression (model_v2)", lr, X_test_sc, y_test)
    joblib.dump({"model": lr, "scaler": scaler}, f"{MODELS_DIR}/model_v2.pkl")
    print(f"Saved: {MODELS_DIR}/model_v2.pkl")


if __name__ == "__main__":
    main()
