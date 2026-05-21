"""
Módulo de discretización del target.

Convierte una variable objetivo cuantitativa (continua) en categórica
para que se pueda usar matriz de confusión y métricas de clasificación.

Métodos disponibles:
- 'cuantiles' : divide en N grupos del mismo tamaño (default 3: bajo/medio/alto)
- 'uniforme'  : divide el rango en N intervalos iguales en valor
- 'binario_mediana' : 0/1 partiendo por la mediana
- 'binario_media'   : 0/1 partiendo por la media
- 'manual'    : cortes definidos por el usuario
"""

import numpy as np
import pandas as pd


def target_es_continuo(serie: pd.Series, umbral_unicos: int = 15) -> bool:
    """
    Decide si una columna se considera continua.
    Heurística:
      - debe ser numérica
      - debe tener más de `umbral_unicos` valores únicos
      - debe tener al menos un valor no entero o un rango amplio
    """
    if not pd.api.types.is_numeric_dtype(serie):
        return False
    n_unicos = serie.dropna().nunique()
    if n_unicos <= umbral_unicos:
        return False
    # Si todos son enteros y el rango es pequeño, probablemente ya es categórica
    s = serie.dropna()
    if (s == s.astype(int)).all() and (s.max() - s.min()) < 20:
        return False
    return True


def discretizar(serie: pd.Series, metodo: str = "cuantiles",
                n_bins: int = 3, cortes: list | None = None,
                etiquetas: list | None = None) -> tuple[pd.Series, str]:
    """
    Aplica la discretización elegida y devuelve la nueva serie + descripción.

    Devuelve:
        nueva_serie   : pd.Series categórica
        descripcion   : string explicativo (para el reporte)
    """
    s = pd.to_numeric(serie, errors="coerce")

    if metodo == "cuantiles":
        labels = etiquetas or _etiquetas_por_defecto(n_bins)
        nueva = pd.qcut(s, q=n_bins, labels=labels, duplicates="drop")
        bordes = pd.qcut(s, q=n_bins, retbins=True, duplicates="drop")[1]
        descripcion = (
            f"Método: CUANTILES en {n_bins} grupos.\n"
            f"Bordes calculados: {[round(b,2) for b in bordes]}\n"
            f"Etiquetas: {list(nueva.cat.categories)}"
        )
    elif metodo == "uniforme":
        labels = etiquetas or _etiquetas_por_defecto(n_bins)
        nueva = pd.cut(s, bins=n_bins, labels=labels, include_lowest=True)
        bordes = pd.cut(s, bins=n_bins, retbins=True, include_lowest=True)[1]
        descripcion = (
            f"Método: INTERVALOS UNIFORMES en {n_bins} grupos.\n"
            f"Bordes: {[round(b,2) for b in bordes]}\n"
            f"Etiquetas: {list(nueva.cat.categories)}"
        )
    elif metodo == "binario_mediana":
        umbral = float(s.median())
        nueva = (s > umbral).astype(int)
        descripcion = (
            f"Método: BINARIZACIÓN por MEDIANA.\n"
            f"Umbral = {umbral:.4f}  ->  0 si valor <= umbral, 1 si valor > umbral"
        )
    elif metodo == "binario_media":
        umbral = float(s.mean())
        nueva = (s > umbral).astype(int)
        descripcion = (
            f"Método: BINARIZACIÓN por MEDIA.\n"
            f"Umbral = {umbral:.4f}  ->  0 si valor <= umbral, 1 si valor > umbral"
        )
    elif metodo == "manual":
        if not cortes or len(cortes) < 2:
            raise ValueError("El método 'manual' requiere lista de cortes")
        labels = etiquetas or _etiquetas_por_defecto(len(cortes) - 1)
        nueva = pd.cut(s, bins=cortes, labels=labels, include_lowest=True)
        descripcion = (
            f"Método: CORTES MANUALES.\n"
            f"Bordes: {cortes}\n"
            f"Etiquetas: {labels}"
        )
    else:
        raise ValueError(f"Método '{metodo}' no reconocido")

    return nueva, descripcion


def _etiquetas_por_defecto(n: int) -> list:
    """Etiquetas para 2, 3, 4 o 5 grupos."""
    if n == 2:
        return ["Bajo", "Alto"]
    if n == 3:
        return ["Bajo", "Medio", "Alto"]
    if n == 4:
        return ["Bajo", "Medio-Bajo", "Medio-Alto", "Alto"]
    if n == 5:
        return ["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"]
    return [f"G{i+1}" for i in range(n)]


def aplicar_discretizacion(df: pd.DataFrame, target: str,
                            reporte: list, metodo: str = "cuantiles",
                            n_bins: int = 3) -> tuple[pd.DataFrame, str]:
    """
    Si el target es continuo lo discretiza y lo escribe en el reporte.
    Devuelve el df modificado y el nombre (posiblemente nuevo) del target.
    """
    if target not in df.columns:
        return df, target

    serie = df[target]
    es_continuo = target_es_continuo(serie)

    reporte.append("=" * 70)
    reporte.append("1.C  DISCRETIZACIÓN DEL TARGET (si aplica)")
    reporte.append("=" * 70)
    reporte.append(f"Variable objetivo: '{target}'")
    reporte.append(f"Tipo detectado: {serie.dtype}")
    reporte.append(f"Valores únicos: {serie.nunique()}")

    if not es_continuo:
        reporte.append("Conclusión: el target YA es categórico/discreto -> no se modifica.")
        reporte.append("")
        return df, target

    reporte.append("Conclusión: target CONTINUO detectado.")
    reporte.append("Se aplica discretización para poder usar matriz de confusión")
    reporte.append("y métricas de clasificación (Accuracy, F1, AUC).")
    reporte.append("")

    nueva_serie, descripcion = discretizar(serie, metodo=metodo, n_bins=n_bins)
    reporte.append(descripcion)

    nuevo_nombre = f"{target}_cat"
    df[nuevo_nombre] = nueva_serie
    df = df.drop(columns=[target])

    distribucion = df[nuevo_nombre].value_counts().sort_index()
    reporte.append("\nDistribución de la nueva variable objetivo:")
    reporte.append(distribucion.to_string())
    reporte.append("")

    print(f"[DISCRETIZACIÓN] Target '{target}' (continuo) -> '{nuevo_nombre}' "
          f"({metodo}, {df[nuevo_nombre].nunique()} clases)")

    return df, nuevo_nombre
