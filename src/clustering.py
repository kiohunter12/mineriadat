"""
Módulo de segmentación (clustering).

Aplica:
- K-Means con método del codo + Silhouette
- Clustering jerárquico aglomerativo + dendrograma + Silhouette
- Compara ambos y propone perfiles de clientes
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin GUI (importante en Visual Studio Code)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


def ejecutar_clustering(X_train_proc, X_train_original, output_dir, reporte,
                        k_min=2, k_max=6):
    """
    Ejecuta K-means y jerárquico sobre los datos ya preprocesados.
    X_train_proc: matriz numérica preprocesada (escalada, encoded)
    X_train_original: dataframe original (para interpretar los perfiles)
    """
    reporte.append("=" * 70)
    reporte.append("3. SEGMENTACIÓN (CLUSTERING)")
    reporte.append("=" * 70)

    # --- Si la matriz tiene demasiadas filas, muestreamos para que jerárquico sea viable
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
    n = X_train_proc.shape[0]
    if n > 1000:
        idx = np.random.RandomState(42).choice(n, 1000, replace=False)
        X_sample = X_train_proc[idx]
        X_orig_sample = X_train_original.iloc[idx].reset_index(drop=True)
        reporte.append(f"(Se muestrean 1000 registros de {n} para clustering jerárquico)")
    else:
        X_sample = X_train_proc
        X_orig_sample = X_train_original.reset_index(drop=True)

    # ============ K-MEANS ============
    reporte.append("\n--- K-MEANS ---")
    reporte.append("Algoritmo de partición que minimiza la distancia intra-cluster.")

    inercias = []
    silhouettes_km = []
    # Limitar k_max al tamaño de la muestra para evitar errores en datasets pequeños
    k_max_efectivo = min(k_max, X_sample.shape[0] - 1)
    rango_k = list(range(k_min, k_max_efectivo + 1))
    if len(rango_k) == 0:
        reporte.append("[!] Dataset demasiado pequeño para clustering. Se omite esta etapa.")
        return {"kmeans_silhouette": None, "kmeans_k": None,
                "jerarquico_silhouette": None, "jerarquico_k": None}
    for k in rango_k:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sample)
        inercias.append(km.inertia_)
        sil = silhouette_score(X_sample, labels)
        silhouettes_km.append(sil)
        reporte.append(f"  k={k}  ->  Silhouette = {sil:.4f}")

    # Mejor k = mayor silhouette
    mejor_k_km = rango_k[int(np.argmax(silhouettes_km))]
    reporte.append(f"  >> Mejor k (K-Means según Silhouette): {mejor_k_km}")

    # Gráfico del codo
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(rango_k, inercias, "o-")
    ax[0].set_xlabel("Número de clusters (k)")
    ax[0].set_ylabel("Inercia")
    ax[0].set_title("Método del Codo - K-Means")
    ax[0].grid(alpha=0.3)

    ax[1].plot(rango_k, silhouettes_km, "o-", color="green")
    ax[1].set_xlabel("Número de clusters (k)")
    ax[1].set_ylabel("Silhouette score")
    ax[1].set_title("Silhouette por k - K-Means")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_kmeans_codo_silhouette.png"), dpi=120)
    plt.close()

    # Ajuste final K-means con el mejor k
    km_final = KMeans(n_clusters=mejor_k_km, random_state=42, n_init=10)
    labels_km = km_final.fit_predict(X_sample)

    # ============ JERÁRQUICO ============
    reporte.append("\n--- CLUSTERING JERÁRQUICO (aglomerativo, enlace 'ward') ---")
    reporte.append("Construye una jerarquía fusionando los puntos más cercanos.")

    Z = linkage(X_sample, method="ward")
    plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8)
    plt.title("Dendrograma - Clustering Jerárquico (Ward)")
    plt.xlabel("Muestras")
    plt.ylabel("Distancia")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_dendrograma.png"), dpi=120)
    plt.close()

    silhouettes_jc = []
    for k in rango_k:
        jc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = jc.fit_predict(X_sample)
        sil = silhouette_score(X_sample, labels)
        silhouettes_jc.append(sil)
        reporte.append(f"  k={k}  ->  Silhouette = {sil:.4f}")

    mejor_k_jc = rango_k[int(np.argmax(silhouettes_jc))]
    reporte.append(f"  >> Mejor k (Jerárquico según Silhouette): {mejor_k_jc}")

    jc_final = AgglomerativeClustering(n_clusters=mejor_k_jc, linkage="ward")
    labels_jc = jc_final.fit_predict(X_sample)

    # ============ INTERPRETACIÓN DE PERFILES ============
    reporte.append("\n--- INTERPRETACIÓN DE LOS CLUSTERS (K-MEANS) ---")
    reporte.append("Promedio de variables numéricas por cluster:")
    df_perfil = X_orig_sample.copy()
    df_perfil["cluster"] = labels_km

    columnas_num = df_perfil.select_dtypes(include="number").columns.tolist()
    columnas_num = [c for c in columnas_num if c != "cluster"]

    if columnas_num:
        resumen = df_perfil.groupby("cluster")[columnas_num].mean().round(2)
        reporte.append(resumen.to_string())

        # Gráfico de heatmap de perfiles
        plt.figure(figsize=(10, max(3, 0.5 * len(columnas_num))))
        # normalización por columna para visualización
        norm = (resumen - resumen.min()) / (resumen.max() - resumen.min() + 1e-9)
        plt.imshow(norm.T, aspect="auto", cmap="RdYlGn")
        plt.colorbar(label="Valor normalizado")
        plt.xticks(range(len(resumen.index)), [f"Cluster {i}" for i in resumen.index])
        plt.yticks(range(len(columnas_num)), columnas_num)
        plt.title("Perfiles de clusters (K-Means)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_perfiles_clusters.png"), dpi=120)
        plt.close()

    reporte.append("")
    reporte.append("PROPUESTA DE PERFILES DE CLIENTES:")
    for c in sorted(set(labels_km)):
        n_c = int((labels_km == c).sum())
        pct = 100 * n_c / len(labels_km)
        reporte.append(f"  Cluster {c}: {n_c} clientes ({pct:.1f}%) - revisar tabla de medias")
        reporte.append(f"    Identificar la(s) variable(s) con valor más alto/bajo")
        reporte.append(f"    para nombrarlo (ej: 'Premium', 'Ocasional', 'Inactivo').")
    reporte.append("")

    return {
        "kmeans_silhouette": max(silhouettes_km),
        "kmeans_k": mejor_k_km,
        "jerarquico_silhouette": max(silhouettes_jc),
        "jerarquico_k": mejor_k_jc,
    }
