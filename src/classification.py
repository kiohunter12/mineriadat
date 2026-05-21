"""
Módulo de clasificación supervisada.

Entrena dos modelos:
- Árbol de Decisión
- Random Forest

Evalúa con: Accuracy, F1-score, AUC.
Todo va dentro de Pipelines para evitar data leakage.
"""

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np


def _calcular_auc(y_true, y_proba):
    """AUC robusto: para binario usa prob clase positiva, para multiclase usa OvR."""
    clases = np.unique(y_true)
    if len(clases) == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    try:
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        return None


def _evaluar(modelo, X, y):
    """Devuelve accuracy, f1 y auc para un conjunto."""
    pred = modelo.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="weighted", zero_division=0)
    try:
        proba = modelo.predict_proba(X)
        auc = _calcular_auc(y, proba)
    except Exception:
        auc = None
    return acc, f1, auc


def entrenar_arbol(preprocesador, X_train, y_train, X_val, y_val, reporte):
    """Árbol de decisión con profundidad limitada para evitar sobreajuste."""
    reporte.append("=" * 70)
    reporte.append("4. CLASIFICACIÓN")
    reporte.append("=" * 70)
    reporte.append("\n--- ÁRBOL DE DECISIÓN ---")
    reporte.append("Modelo interpretable que divide el espacio con reglas if-else.")

    modelo = Pipeline([
        ("prep", preprocesador),
        ("clf", DecisionTreeClassifier(max_depth=6, random_state=42)),
    ])
    modelo.fit(X_train, y_train)

    acc, f1, auc = _evaluar(modelo, X_val, y_val)
    reporte.append(f"  Validación  -> Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: "
                   f"{auc:.4f}" if auc else f"  Validación -> Acc: {acc:.4f}  F1: {f1:.4f}")
    return modelo, {"modelo": "Árbol de Decisión", "accuracy": acc, "f1": f1, "auc": auc}


def entrenar_random_forest(preprocesador, X_train, y_train, X_val, y_val, reporte):
    """Random Forest: ensamble de árboles con bagging."""
    reporte.append("\n--- RANDOM FOREST ---")
    reporte.append("Ensamble de árboles que reduce varianza y suele dar mejor desempeño.")

    modelo = Pipeline([
        ("prep", preprocesador),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=None,
                                        random_state=42, n_jobs=-1)),
    ])
    modelo.fit(X_train, y_train)

    acc, f1, auc = _evaluar(modelo, X_val, y_val)
    reporte.append(f"  Validación  -> Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: "
                   f"{auc:.4f}" if auc else f"  Validación -> Acc: {acc:.4f}  F1: {f1:.4f}")
    return modelo, {"modelo": "Random Forest", "accuracy": acc, "f1": f1, "auc": auc}


def evaluar_en_test(modelos: dict, X_test, y_test, reporte):
    """Evalúa cada modelo en el conjunto de prueba (intacto hasta este punto)."""
    reporte.append("\n--- EVALUACIÓN EN CONJUNTO DE PRUEBA (test) ---")
    resultados_test = []
    for nombre, modelo in modelos.items():
        acc, f1, auc = _evaluar(modelo, X_test, y_test)
        auc_txt = f"{auc:.4f}" if auc is not None else "N/A"
        reporte.append(f"  {nombre:25s} -> Acc: {acc:.4f}  F1: {f1:.4f}  AUC: {auc_txt}")
        resultados_test.append({
            "modelo": nombre, "accuracy": acc, "f1": f1, "auc": auc,
        })
    reporte.append("")
    return resultados_test
