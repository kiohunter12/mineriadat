"""
Módulo de carga y exploración de datos.
Funciona con cualquier CSV: detecta tipos de columnas automáticamente.
"""

import pandas as pd
import numpy as np


def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """Carga un CSV intentando distintos separadores y encodings."""
    intentos = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin-1"},
        {"sep": ";", "encoding": "latin-1"},
    ]
    ultimo_error = None
    for cfg in intentos:
        try:
            df = pd.read_csv(ruta_csv, **cfg)
            if df.shape[1] > 1:
                print(f"[OK] Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
                print(f"     Separador='{cfg['sep']}'  Encoding='{cfg['encoding']}'")
                return df
        except Exception as e:
            ultimo_error = e
    raise RuntimeError(f"No se pudo leer el CSV: {ultimo_error}")


def detectar_objetivo(df: pd.DataFrame, target: str | None) -> str:
    """
    Determina la columna objetivo.
    - Si el usuario la pasa por argumento, la usa.
    - Si no, busca columnas típicas (target, y, clase, respuesta, etc.).
    - Si tampoco encuentra, toma la última columna del dataframe.
    """
    if target and target in df.columns:
        print(f"[OK] Variable objetivo definida por el usuario: '{target}'")
        return target

    candidatos = ["target", "y", "clase", "class", "label", "respuesta",
                  "response", "churn", "objetivo", "etiqueta"]
    for col in df.columns:
        if col.lower() in candidatos:
            print(f"[OK] Variable objetivo detectada automáticamente: '{col}'")
            return col

    target = df.columns[-1]
    print(f"[!] No se especificó variable objetivo. Se usa la última columna: '{target}'")
    return target


def clasificar_columnas(df: pd.DataFrame, target: str) -> tuple[list, list]:
    """Separa columnas numéricas y categóricas (excluyendo la objetivo)."""
    features = [c for c in df.columns if c != target]
    numericas = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df[features].select_dtypes(exclude=[np.number]).columns.tolist()
    return numericas, categoricas


def explorar(df: pd.DataFrame, target: str, reporte: list) -> None:
    """Realiza la exploración inicial y va escribiendo en el reporte."""
    reporte.append("=" * 70)
    reporte.append("1. EXPLORACIÓN INICIAL DEL DATASET")
    reporte.append("=" * 70)
    reporte.append(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    reporte.append(f"Variable objetivo: {target}")

    numericas, categoricas = clasificar_columnas(df, target)
    reporte.append(f"Columnas numéricas ({len(numericas)}): {numericas}")
    reporte.append(f"Columnas categóricas ({len(categoricas)}): {categoricas}")

    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if len(nulos) > 0:
        reporte.append(f"\nValores nulos detectados:\n{nulos.to_string()}")
    else:
        reporte.append("\nNo hay valores nulos en el dataset.")

    reporte.append(f"\nDistribución de la variable objetivo '{target}':")
    reporte.append(df[target].value_counts().to_string())
    reporte.append("")
