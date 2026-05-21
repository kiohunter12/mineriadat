"""
Módulo de partición de datos y modelo baseline.

- Partición estratificada 60/20/20 (train/val/test).
- Preprocesamiento dentro de Pipeline para EVITAR DATA LEAKAGE:
  el imputador, el escalador y el OneHotEncoder se ajustan SOLO con train
  y luego se aplican a val y test.
- Baseline = DummyClassifier (predice la clase mayoritaria).
"""

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score


def particionar(df, target, reporte):
    """
    Divide el dataset en 60% train, 20% validación y 20% test.
    Mantiene la proporción de clases (stratify).
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Primero separamos test (20%) del resto
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    # Del 80% restante, sacamos validación (25% de 80% = 20% del total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    reporte.append("=" * 70)
    reporte.append("2. PARTICIÓN DE DATOS Y MODELO BASELINE")
    reporte.append("=" * 70)
    reporte.append("Se aplica partición ESTRATIFICADA 60/20/20:")
    reporte.append(f"  - Entrenamiento: {X_train.shape[0]} registros (60%)")
    reporte.append(f"  - Validación:    {X_val.shape[0]} registros (20%)")
    reporte.append(f"  - Prueba:        {X_test.shape[0]} registros (20%)")
    reporte.append("")
    reporte.append("¿QUÉ ES DATA LEAKAGE?")
    reporte.append("Es la fuga de información del conjunto de prueba hacia el modelo")
    reporte.append("durante el entrenamiento. Provoca métricas falsamente altas que no")
    reporte.append("se reproducen en producción.")
    reporte.append("")
    reporte.append("CÓMO LO EVITAMOS EN ESTE PROYECTO:")
    reporte.append("  1) Stratify mantiene la proporción de clases sin sesgar el split.")
    reporte.append("  2) Imputación, escalado y OneHotEncoder van DENTRO de un Pipeline")
    reporte.append("     que se ajusta SOLO con X_train; val y test solo se transforman.")
    reporte.append("  3) El conjunto de test no se toca hasta el final.")
    reporte.append("")

    return X_train, X_val, X_test, y_train, y_val, y_test


def construir_preprocesador(X_train):
    """
    Construye un ColumnTransformer que:
    - imputa+escala columnas numéricas
    - imputa+OneHotEncode columnas categóricas
    Se ajusta una sola vez con X_train.
    """
    numericas = X_train.select_dtypes(include="number").columns.tolist()
    categoricas = X_train.select_dtypes(exclude="number").columns.tolist()

    pipe_num = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
    ])
    pipe_cat = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", pipe_num, numericas),
        ("cat", pipe_cat, categoricas),
    ])


def baseline(X_train, X_val, y_train, y_val, reporte):
    """
    Modelo baseline: DummyClassifier que predice la clase mayoritaria.
    Sirve como mínimo a superar por cualquier modelo real.
    """
    modelo = DummyClassifier(strategy="most_frequent", random_state=42)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_val)

    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred, average="weighted", zero_division=0)

    reporte.append("MODELO BASELINE (DummyClassifier - clase mayoritaria):")
    reporte.append(f"  Accuracy en validación: {acc:.4f}")
    reporte.append(f"  F1-score en validación: {f1:.4f}")
    reporte.append("Cualquier modelo entrenado debe superar estas métricas para")
    reporte.append("considerarse útil.")
    reporte.append("")

    return {"modelo": "Baseline (Dummy)", "accuracy": acc, "f1": f1, "auc": None}
