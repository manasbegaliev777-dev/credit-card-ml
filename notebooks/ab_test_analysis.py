"""
A/B тест: RandomForest (model_v1) vs LogisticRegression (model_v2)
Разделение клиентов 50/50 по MD5-хэшу от ID.
"""

import hashlib
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Конфиг ────────────────────────────────────────────────────────────────────
DATA_PATH     = "UCI_Credit_Card.csv"
MODEL_V1_PATH = "models/model_v1.pkl"
MODEL_V2_PATH = "models/model_v2.pkl"
TARGET        = "default.payment.next.month"
Z_95          = 1.96


# ── Загрузка ──────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def load_models():
    v1 = joblib.load(MODEL_V1_PATH)
    v2 = joblib.load(MODEL_V2_PATH)   # {"model": lr, "scaler": scaler}
    return v1, v2


# ── Роутинг по MD5 ────────────────────────────────────────────────────────────
def assign_group(client_id) -> str:
    h = int(hashlib.md5(str(client_id).encode()).hexdigest(), 16)
    return "A" if h % 2 == 0 else "B"


# ── Предсказания ──────────────────────────────────────────────────────────────
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def predict_group(X, y_true, model_v1, model_v2_bundle, groups):
    mask_a = groups == "A"
    mask_b = groups == "B"

    # Группа A — model_v1
    ya_pred = model_v1.predict(X[mask_a])

    # Группа B — model_v2 (с масштабированием)
    scaler  = model_v2_bundle["scaler"]
    model   = model_v2_bundle["model"]
    yb_pred = model.predict(scaler.transform(X[mask_b]))

    return (
        y_true[mask_a], ya_pred,
        y_true[mask_b], yb_pred,
    )


# ── Z-test для двух долей ────────────────────────────────────────────────────
def z_test(p1, p2, n1, n2):
    """Односторонний Z-test: H1: p2 > p1"""
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    # p-value (одностороннее): P(Z > z)
    # Аппроксимация через CDF нормального распределения
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    return z, p_value


# ── Доверительные интервалы ───────────────────────────────────────────────────
def confidence_interval(p, n, z=Z_95):
    """CI = p ± Z * sqrt(p(1-p)/n)"""
    margin = z * math.sqrt(p * (1 - p) / n)
    return round(p - margin, 4), round(p + margin, 4)


# ── Метрики ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "n":         len(y_true),
    }


# ── Главная функция ───────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  A/B TEST: RandomForest (v1) vs LogisticRegression (v2)")
    print("=" * 60)

    # Загрузка данных
    df       = load_data()
    ids      = df["ID"].values
    X        = df.drop(columns=["ID", TARGET]).values
    y        = df[TARGET].values
    groups   = np.array([assign_group(i) for i in ids])

    print(f"\nВсего клиентов : {len(df):,}")
    print(f"Группа A (v1)  : {(groups == 'A').sum():,} ({(groups == 'A').mean()*100:.1f}%)")
    print(f"Группа B (v2)  : {(groups == 'B').sum():,} ({(groups == 'B').mean()*100:.1f}%)")

    # Загрузка моделей
    model_v1, model_v2_bundle = load_models()

    # Предсказания
    ya_true, ya_pred, yb_true, yb_pred = predict_group(
        X, y, model_v1, model_v2_bundle, groups
    )

    # Метрики
    metrics_a = compute_metrics(ya_true, ya_pred)
    metrics_b = compute_metrics(yb_true, yb_pred)

    print("\n--- Метрики ---")
    print(f"{'Метрика':<12} {'Группа A (v1)':>15} {'Группа B (v2)':>15}")
    print("-" * 44)
    for key in ["F1", "Precision", "Recall"]:
        print(f"{key:<12} {metrics_a[key]:>15.4f} {metrics_b[key]:>15.4f}")
    print(f"{'n':12} {metrics_a['n']:>15,} {metrics_b['n']:>15,}")

    # Доверительные интервалы для F1
    ci_a = confidence_interval(metrics_a["F1"], metrics_a["n"])
    ci_b = confidence_interval(metrics_b["F1"], metrics_b["n"])

    print("\n--- Доверительные интервалы F1 (95%) ---")
    print(f"  Группа A (v1): {ci_a[0]} — {ci_a[1]}")
    print(f"  Группа B (v2): {ci_b[0]} — {ci_b[1]}")

    # Z-test по F1
    z_stat, p_val = z_test(
        metrics_a["F1"], metrics_b["F1"],
        metrics_a["n"],  metrics_b["n"]
    )

    print("\n--- Z-test (H1: F1_B > F1_A) ---")
    print(f"  Z-статистика : {z_stat:.4f}")
    print(f"  p-value      : {p_val:.4f}")
    print(f"  Значимость   : {'ДА (p < 0.05)' if p_val < 0.05 else 'НЕТ (p >= 0.05)'}")

    # Итог
    delta = metrics_b["F1"] - metrics_a["F1"]
    improvement_pct = (delta / metrics_a["F1"] * 100) if metrics_a["F1"] > 0 else 0

    print("\n--- Итоговый вывод ---")
    if p_val < 0.05 and improvement_pct >= 5:
        winner = "B (model_v2 — LogisticRegression)"
        reason = f"F1 выше на {improvement_pct:.1f}%, p-value={p_val:.4f} < 0.05"
    elif p_val < 0.05 and delta > 0:
        winner = "B (model_v2) — незначительное улучшение"
        reason = f"F1 выше на {improvement_pct:.1f}%, но < 5% порога"
    else:
        winner = "A (model_v1 — RandomForest)"
        reason = f"F1 выше или нет статистически значимой разницы (p={p_val:.4f})"

    print(f"  Победитель   : {winner}")
    print(f"  Причина      : {reason}")
    print(f"  Разница F1   : {delta:+.4f} ({improvement_pct:+.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
