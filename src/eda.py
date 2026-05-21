"""
Módulo de EDA (Análisis Exploratorio) con limpieza automática.

Detecta y corrige problemas comunes en los datos ANTES de entrenar:
1. Marcadores no estándar de nulos: 'N/A', 'na', '?', '-', 'null', '', '#N/A', etc.
2. Espacios sobrantes en strings (' Lima ' -> 'Lima').
3. Strings que en realidad son números mal formateados:
   - Coma decimal: '12,5'        -> 12.5
   - Separador miles: '1,500'    -> 1500
   - Formato europeo: '1.234,56' -> 1234.56
   - Símbolos de moneda: 'S/. 1500', '$1,500', '€ 200'
   - Porcentajes: '25%' -> 25
4. Duplicados.
5. Reporta TODO lo que cambió para auditoría.
"""

import re
import pandas as pd
import numpy as np


# Cadenas que típicamente representan nulos pero no son np.nan reales
MARCADORES_NULOS = {
    "", " ", "NA", "N/A", "na", "n/a", "NULL", "null", "None", "none",
    "?", "-", "--", "#N/A", "#NULL!", "NaN", "nan", ".", "S/I", "s/i",
    "sin dato", "Sin Dato", "SIN DATO", "desconocido", "Desconocido",
}


def _intentar_a_numero(valor):
    """
    Intenta convertir un string a número manejando los formatos comunes.
    Devuelve float si puede, np.nan si no.
    """
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip()
    if s == "":
        return np.nan

    # Detectar y manejar % (lo conservamos como número, ej '25%' -> 25)
    es_porcentaje = s.endswith("%")
    if es_porcentaje:
        s = s[:-1].strip()

    # Quitar símbolos de moneda comunes y espacios
    s = re.sub(r"[\$€¥£]", "", s)
    s = re.sub(r"S/\.?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"USD|PEN|EUR|GBP", "", s, flags=re.IGNORECASE)
    s = s.strip()

    if s == "" or s == "-":
        return np.nan

    # Manejar separadores:
    #   "1,234.56" -> US: coma=miles, punto=decimal -> 1234.56
    #   "1.234,56" -> EU: punto=miles, coma=decimal -> 1234.56
    #   "12,5"     -> coma decimal -> 12.5
    #   "1,500"    -> coma como miles (3 dígitos después) -> 1500
    tiene_coma = "," in s
    tiene_punto = "." in s

    if tiene_coma and tiene_punto:
        # El último separador es el decimal
        if s.rfind(",") > s.rfind("."):
            # formato europeo: 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:
            # formato US: 1,234.56
            s = s.replace(",", "")
    elif tiene_coma:
        partes = s.split(",")
        # Si después de la coma hay exactamente 3 dígitos -> es separador de miles
        if len(partes) == 2 and len(partes[1]) == 3 and partes[1].isdigit():
            s = s.replace(",", "")
        else:
            # Coma decimal
            s = s.replace(",", ".")

    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def limpiar_dataset(df: pd.DataFrame, reporte: list) -> pd.DataFrame:
    """
    Aplica limpieza automática al dataframe.
    Modifica df y va escribiendo los cambios en `reporte`.
    """
    cambios = []
    df = df.copy()

    reporte.append("=" * 70)
    reporte.append("1.B  EDA + LIMPIEZA AUTOMÁTICA")
    reporte.append("=" * 70)

    # ---- 1) Quitar espacios en blanco y unificar marcadores de nulo ----
    cols_objeto = df.select_dtypes(include="object").columns.tolist()
    nulos_recuperados = 0
    for col in cols_objeto:
        try:
            df[col] = df[col].astype(str).str.strip()
        except Exception:
            pass
        antes = df[col].isnull().sum()
        df[col] = df[col].replace(list(MARCADORES_NULOS), np.nan)
        despues = df[col].isnull().sum()
        if despues > antes:
            n = int(despues - antes)
            nulos_recuperados += n
            cambios.append(f"  - '{col}': {n} valores tipo 'N/A','?','-','' convertidos a NaN")

    # ---- 2) Detectar columnas object que en realidad son numéricas ----
    columnas_convertidas = []
    for col in cols_objeto:
        muestra = df[col].dropna().astype(str).head(80)
        if len(muestra) == 0:
            continue
        convertidos = muestra.apply(_intentar_a_numero)
        tasa_exito = convertidos.notna().sum() / len(muestra)

        # Si más del 80% de los valores no nulos se pueden convertir -> es numérica
        if tasa_exito >= 0.80:
            df[col] = df[col].apply(_intentar_a_numero)
            columnas_convertidas.append(col)
            cambios.append(
                f"  - '{col}': reconvertida a numérica "
                f"(éxito {tasa_exito*100:.0f}%) - antes era texto"
            )

    # ---- 3) Detectar y quitar duplicados exactos ----
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        cambios.append(f"  - Filas duplicadas eliminadas: {n_dup}")

    # ---- 4) Tipos finales ----
    reporte.append(f"\nMarcadores no estándar de NaN convertidos: {nulos_recuperados}")
    reporte.append(f"Columnas reconvertidas a numéricas: {len(columnas_convertidas)}  "
                   f"{columnas_convertidas if columnas_convertidas else ''}")
    reporte.append(f"Filas duplicadas eliminadas: {n_dup}")

    reporte.append("\nDetalle de cambios:")
    if cambios:
        reporte.extend(cambios)
    else:
        reporte.append("  No fueron necesarios ajustes (los datos ya estaban limpios).")

    # ---- 5) Resumen de tipos por columna después de la limpieza ----
    reporte.append("\nTipos de columna después de la limpieza:")
    tipos = pd.DataFrame({
        "columna": df.columns,
        "tipo": df.dtypes.astype(str).values,
        "nulos": df.isnull().sum().values,
        "%nulos": (df.isnull().mean() * 100).round(1).values,
    })
    reporte.append(tipos.to_string(index=False))
    reporte.append("")

    print(f"[EDA] Limpieza completa. {len(cambios)} ajustes aplicados.")
    return df
