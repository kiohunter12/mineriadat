"""
Módulo de comparación entre dos datasets.

Recibe los resultados de haber corrido el pipeline en dos datasets
independientes y genera:
- Tabla comparativa lado a lado
- Gráfico de barras comparando métricas
- Reporte interpretativo de las diferencias
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def comparar_datasets(resultado1: dict, resultado2: dict, output_dir: str):
    """
    Cada 'resultado' es un dict con:
        - nombre: str (etiqueta del dataset)
        - tabla: DataFrame con columnas modelo, conjunto, accuracy, f1, auc
        - clustering: dict con métricas de silhouette
        - filas, columnas: tamaño del dataset
        - target: nombre de la variable objetivo
    """
    os.makedirs(output_dir, exist_ok=True)
    lineas = []

    lineas.append("=" * 70)
    lineas.append("COMPARACIÓN ENTRE DATASETS")
    lineas.append("=" * 70)
    lineas.append(f"Dataset A: {resultado1['nombre']}  ({resultado1['filas']} filas x "
                  f"{resultado1['columnas']} cols, target='{resultado1['target']}')")
    lineas.append(f"Dataset B: {resultado2['nombre']}  ({resultado2['filas']} filas x "
                  f"{resultado2['columnas']} cols, target='{resultado2['target']}')")
    lineas.append("")

    # ===== 1. Comparación de clustering =====
    lineas.append("-" * 70)
    lineas.append("1) SEGMENTACIÓN (CLUSTERING)")
    lineas.append("-" * 70)
    c1 = resultado1["clustering"]
    c2 = resultado2["clustering"]
    df_clust = pd.DataFrame({
        "Métrica": ["Mejor k (K-Means)", "Silhouette K-Means",
                    "Mejor k (Jerárquico)", "Silhouette Jerárquico"],
        resultado1["nombre"]: [c1["kmeans_k"], round(c1["kmeans_silhouette"], 4),
                                c1["jerarquico_k"], round(c1["jerarquico_silhouette"], 4)],
        resultado2["nombre"]: [c2["kmeans_k"], round(c2["kmeans_silhouette"], 4),
                                c2["jerarquico_k"], round(c2["jerarquico_silhouette"], 4)],
    })
    lineas.append(df_clust.to_string(index=False))
    lineas.append("")

    # ===== 2. Comparación de clasificación (resultados en TEST) =====
    lineas.append("-" * 70)
    lineas.append("2) CLASIFICACIÓN - MÉTRICAS EN TEST")
    lineas.append("-" * 70)

    t1 = resultado1["tabla"]
    t2 = resultado2["tabla"]
    t1_test = t1[t1["conjunto"] == "Test"].copy()
    t2_test = t2[t2["conjunto"] == "Test"].copy()
    t1_test["dataset"] = resultado1["nombre"]
    t2_test["dataset"] = resultado2["nombre"]

    comp = pd.concat([t1_test, t2_test], ignore_index=True)
    comp = comp[["dataset", "modelo", "accuracy", "f1", "auc"]]
    lineas.append(comp.to_string(index=False))
    lineas.append("")

    # Guardar CSV
    comp.to_csv(os.path.join(output_dir, "comparacion_datasets.csv"), index=False)

    # ===== 3. Gráfico de barras =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metricas = ["accuracy", "f1", "auc"]
    titulos = ["Accuracy", "F1-score", "AUC"]
    modelos = comp["modelo"].unique()

    for ax, m, t in zip(axes, metricas, titulos):
        x = np.arange(len(modelos))
        ancho = 0.35
        vals1 = [t1_test[t1_test["modelo"] == mod][m].values[0] for mod in modelos]
        vals2 = [t2_test[t2_test["modelo"] == mod][m].values[0] for mod in modelos]
        # Reemplazar NaN por 0 para graficar
        vals1 = [0 if pd.isna(v) else v for v in vals1]
        vals2 = [0 if pd.isna(v) else v for v in vals2]

        ax.bar(x - ancho/2, vals1, ancho, label=resultado1["nombre"], color="#3b82f6")
        ax.bar(x + ancho/2, vals2, ancho, label=resultado2["nombre"], color="#f59e0b")
        ax.set_xticks(x)
        ax.set_xticklabels(modelos, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(t)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparacion_metricas.png"), dpi=120)
    plt.close()

    # ===== 4. Interpretación automática =====
    lineas.append("-" * 70)
    lineas.append("3) INTERPRETACIÓN")
    lineas.append("-" * 70)

    # Mejor dataset según F1 promedio (excluyendo baseline)
    def f1_promedio(t):
        candidatos = t[~t["modelo"].str.lower().str.contains("baseline|dummy")]
        return candidatos["f1"].mean()

    f1_a = f1_promedio(t1_test)
    f1_b = f1_promedio(t2_test)

    if f1_a > f1_b:
        ganador, perdedor = resultado1["nombre"], resultado2["nombre"]
        diff = f1_a - f1_b
    else:
        ganador, perdedor = resultado2["nombre"], resultado1["nombre"]
        diff = f1_b - f1_a

    lineas.append(f"- En clasificación, los modelos rinden mejor en '{ganador}' que en")
    lineas.append(f"  '{perdedor}' (diferencia promedio de F1 = {diff:.4f}).")

    # Comparar silhouette
    s_a = c1["kmeans_silhouette"]
    s_b = c2["kmeans_silhouette"]
    if s_a > s_b:
        lineas.append(f"- En clustering, '{resultado1['nombre']}' presenta clusters mejor")
        lineas.append(f"  separados (Silhouette = {s_a:.3f} vs {s_b:.3f}).")
    else:
        lineas.append(f"- En clustering, '{resultado2['nombre']}' presenta clusters mejor")
        lineas.append(f"  separados (Silhouette = {s_b:.3f} vs {s_a:.3f}).")

    lineas.append("")
    lineas.append("POSIBLES EXPLICACIONES DE LAS DIFERENCIAS:")
    lineas.append("  - Distinta calidad de los datos (nulos, ruido, balance de clases).")
    lineas.append("  - Distinto poder predictivo de las variables disponibles.")
    lineas.append("  - Distinto tamaño muestral (más datos suelen ayudar al modelo).")
    lineas.append("  - Distinta dificultad intrínseca del problema en cada dataset.")
    lineas.append("")

    # Guardar reporte
    ruta = os.path.join(output_dir, "comparacion_datasets.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    print("\n".join(lineas))
    print(f"\n[OK] Comparación guardada en: {output_dir}")
