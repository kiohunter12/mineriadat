"""
========================================================================
TRABAJO UNIDAD 2 - MINERIA DE DATOS
Pipeline: EDA + Limpieza + Discretizacion + Particion + Baseline +
          Clustering + Clasificacion + Evaluacion + Matriz de Confusion

Soporta 1 o 2 datasets y discretizacion automatica de target continuo.
========================================================================
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from src.data_loader import cargar_datos, detectar_objetivo, explorar
from src.eda import limpiar_dataset
from src.discretizacion import aplicar_discretizacion
from src.partition import particionar, construir_preprocesador, baseline
from src.clustering import ejecutar_clustering
from src.classification import entrenar_arbol, entrenar_random_forest, evaluar_en_test
from src.evaluation import (curva_roc, tabla_comparativa, matriz_confusion,
                             resumen_ejecutivo)
from src.comparar import comparar_datasets


def ejecutar_pipeline(ruta_csv, target, output_dir, etiqueta="",
                       metodo_discretizacion="cuantiles", n_bins=3):
    """Ejecuta el pipeline completo sobre un dataset."""
    os.makedirs(output_dir, exist_ok=True)
    reporte = []
    titulo = f"REPORTE DE EJECUCION - {etiqueta}" if etiqueta else "REPORTE DE EJECUCION"
    reporte.append(titulo)
    reporte.append(f"Archivo: {ruta_csv}")
    reporte.append("")

    df = cargar_datos(ruta_csv)
    df = limpiar_dataset(df, reporte)
    target = detectar_objetivo(df, target)
    # Si el target es continuo, lo discretizamos antes de la exploracion
    df, target = aplicar_discretizacion(df, target, reporte,
                                         metodo=metodo_discretizacion,
                                         n_bins=n_bins)
    explorar(df, target, reporte)

    X_train, X_val, X_test, y_train, y_val, y_test = particionar(df, target, reporte)
    preprocesador = construir_preprocesador(X_train)
    metricas_baseline = baseline(X_train, X_val, y_train, y_val, reporte)

    preprocesador_clust = construir_preprocesador(X_train)
    X_train_proc = preprocesador_clust.fit_transform(X_train)
    metricas_clust = ejecutar_clustering(X_train_proc, X_train, output_dir, reporte)

    modelo_arbol, met_arbol = entrenar_arbol(
        preprocesador, X_train, y_train, X_val, y_val, reporte)
    modelo_rf, met_rf = entrenar_random_forest(
        construir_preprocesador(X_train), X_train, y_train, X_val, y_val, reporte)

    modelos = {
        "Arbol de Decision": modelo_arbol,
        "Random Forest": modelo_rf,
    }
    resultados_test = evaluar_en_test(modelos, X_test, y_test, reporte)

    resultados_val = [metricas_baseline, met_arbol, met_rf]
    curva_roc(modelos, X_test, y_test, output_dir, reporte)
    tabla = tabla_comparativa(resultados_val, resultados_test, output_dir, reporte)

    test_df = pd.DataFrame(resultados_test)
    mejor_idx = test_df["f1"].idxmax()
    mejor_nombre = test_df.loc[mejor_idx, "modelo"]
    mejor_modelo = modelos[mejor_nombre]

    matriz_confusion(mejor_nombre, mejor_modelo, X_test, y_test, output_dir, reporte)
    resumen_ejecutivo(tabla, reporte)

    ruta_reporte = os.path.join(output_dir, "reporte_resultados.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write("\n".join(reporte))

    print(f"\n[OK] Resultados guardados en: {os.path.abspath(output_dir)}")

    return {
        "nombre": etiqueta or os.path.basename(ruta_csv),
        "tabla": tabla,
        "clustering": metricas_clust,
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "target": target,
    }


def modo_interactivo():
    """Pregunta al usuario cuantos datasets y sus rutas."""
    print("\n" + "=" * 70)
    print("  PIPELINE DE MINERIA DE DATOS - UNIDAD 2")
    print("=" * 70)
    print("\nCuantos datasets vas a analizar?")
    print("  1 -> Procesar un solo dataset (flujo completo)")
    print("  2 -> Procesar dos datasets y compararlos\n")

    while True:
        opcion = input("Tu eleccion [1/2]: ").strip()
        if opcion in ("1", "2"):
            break
        print("  >> Ingresa 1 o 2")

    archivos = []
    n = int(opcion)
    for i in range(1, n + 1):
        ruta = input(f"\nRuta del dataset {i} (Enter usa el de ejemplo): ").strip()
        if not ruta:
            ruta = "data/dataset_ejemplo.csv" if i == 1 else "data/dataset_ejemplo2.csv"
        target = input(f"Variable objetivo del dataset {i} (Enter para autodetectar): ").strip()
        archivos.append((ruta, target or None))
    return archivos


def main():
    parser = argparse.ArgumentParser(description="Pipeline de mineria de datos")
    parser.add_argument("--data", default=None, help="Ruta al primer CSV")
    parser.add_argument("--data2", default=None, help="Ruta al segundo CSV (opcional)")
    parser.add_argument("--target", default=None, help="Columna objetivo del primer dataset")
    parser.add_argument("--target2", default=None, help="Columna objetivo del segundo dataset")
    parser.add_argument("--output", default="output", help="Carpeta base de resultados")
    parser.add_argument("--interactivo", action="store_true", help="Forzar menu interactivo")
    parser.add_argument("--discretizar", default="cuantiles",
                        choices=["cuantiles", "uniforme", "binario_mediana",
                                 "binario_media"],
                        help="Metodo para discretizar target continuo")
    parser.add_argument("--n_bins", type=int, default=3,
                        help="Numero de grupos para discretizar (default 3)")
    args = parser.parse_args()

    if args.interactivo or (args.data is None and args.data2 is None):
        archivos = modo_interactivo()
    else:
        archivos = [(args.data, args.target)]
        if args.data2:
            archivos.append((args.data2, args.target2))

    os.makedirs(args.output, exist_ok=True)

    if len(archivos) == 1:
        ruta, target = archivos[0]
        ejecutar_pipeline(ruta, target, args.output, etiqueta=os.path.basename(ruta),
                           metodo_discretizacion=args.discretizar, n_bins=args.n_bins)
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETADO (1 dataset)")
        print("=" * 70)
    else:
        resultados = []
        for i, (ruta, target) in enumerate(archivos, start=1):
            print(f"\n\n{'#' * 70}")
            print(f"  PROCESANDO DATASET {i}: {ruta}")
            print(f"{'#' * 70}")
            sub_output = os.path.join(args.output, f"dataset_{i}")
            etiqueta = os.path.splitext(os.path.basename(ruta))[0]
            res = ejecutar_pipeline(ruta, target, sub_output, etiqueta=etiqueta,
                                     metodo_discretizacion=args.discretizar,
                                     n_bins=args.n_bins)
            resultados.append(res)

        print(f"\n\n{'#' * 70}")
        print("  COMPARACION ENTRE LOS DOS DATASETS")
        print(f"{'#' * 70}")
        comparar_datasets(resultados[0], resultados[1],
                          os.path.join(args.output, "comparacion"))

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETADO (2 datasets + comparacion)")
        print("=" * 70)
        print(f"  - {args.output}/dataset_1/   -> analisis del primer dataset")
        print(f"  - {args.output}/dataset_2/   -> analisis del segundo dataset")
        print(f"  - {args.output}/comparacion/ -> tabla y grafico comparativo")


if __name__ == "__main__":
    main()
