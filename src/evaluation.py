"""
Modulo de evaluacion comparativa.

- Curvas ROC superpuestas para todos los modelos
- Tabla comparativa de corridas
- Matriz de confusion + interpretacion enriquecida:
    * conteos + matriz normalizada
    * metricas extendidas (Precision, Recall, Specificity, NPV, F1,
      Balanced Accuracy, MCC, Kappa)
    * interpretacion por clase en multiclase
    * analisis de confusiones (que clase se mezcla con cual)
    * recomendaciones automaticas segun el patron de errores
    * narrativa en lenguaje natural
- Resumen ejecutivo no tecnico
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score, precision_score, recall_score, f1_score
)


def _es_binario(y):
    return len(np.unique(y)) == 2


def curva_roc(modelos: dict, X_test, y_test, output_dir, reporte):
    """Genera curvas ROC para clasificacion binaria."""
    if not _es_binario(y_test):
        reporte.append("(Curva ROC se omite: problema multiclase. Se reporta AUC weighted.)")
        return

    reporte.append("=" * 70)
    reporte.append("5. EVALUACION COMPARATIVA")
    reporte.append("=" * 70)
    reporte.append("\n--- CURVAS ROC ---")
    reporte.append("La curva ROC muestra el equilibrio entre verdaderos positivos y")
    reporte.append("falsos positivos. Mientras mas cercana a la esquina superior izquierda,")
    reporte.append("mejor el modelo. El AUC resume la curva en un solo numero (0.5 = azar).")

    plt.figure(figsize=(8, 6))
    clase_pos = np.unique(y_test)[1]
    for nombre, modelo in modelos.items():
        try:
            proba = modelo.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba, pos_label=clase_pos)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{nombre} (AUC = {roc_auc:.3f})")
            reporte.append(f"  {nombre}: AUC = {roc_auc:.4f}")
        except Exception as e:
            reporte.append(f"  {nombre}: no se pudo calcular ROC ({e})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Azar (AUC=0.5)")
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title("Curvas ROC comparativas")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_curvas_roc.png"), dpi=120)
    plt.close()


def tabla_comparativa(resultados_val, resultados_test, output_dir, reporte):
    """Construye y guarda la tabla comparativa de modelos."""
    reporte.append("\n--- TABLA COMPARATIVA DE MODELOS ---")

    df_val = pd.DataFrame(resultados_val)
    df_val["conjunto"] = "Validacion"
    df_test = pd.DataFrame(resultados_test)
    df_test["conjunto"] = "Test"

    tabla = pd.concat([df_val, df_test], ignore_index=True)
    tabla = tabla[["modelo", "conjunto", "accuracy", "f1", "auc"]]
    tabla.to_csv(os.path.join(output_dir, "tabla_comparativa.csv"), index=False)

    reporte.append(tabla.to_string(index=False))
    reporte.append("")
    return tabla


# ===========================================================================
# MATRIZ DE CONFUSION ENRIQUECIDA
# ===========================================================================
def _generar_imagenes_matriz(cm, cm_norm, clases, modelo_nombre, output_dir):
    """Genera dos imagenes: la matriz de conteos y la normalizada."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # Conteos
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    disp.plot(ax=ax[0], cmap="Blues", values_format="d", colorbar=False)
    ax[0].set_title(f"Conteos absolutos - {modelo_nombre}")

    # Normalizada por fila (recall por clase)
    disp_n = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=clases)
    disp_n.plot(ax=ax[1], cmap="Greens", values_format=".2f", colorbar=False)
    ax[1].set_title("Normalizada por fila (% de cada clase real)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_matriz_confusion.png"), dpi=120)
    plt.close()


def _metricas_binarias(cm):
    """Calcula las metricas clasicas binarias."""
    tn, fp, fn, tp = cm.ravel()
    eps = 1e-9
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Precision":   tp / (tp + fp + eps),
        "Recall":      tp / (tp + fn + eps),   # Sensibilidad
        "Specificity": tn / (tn + fp + eps),
        "NPV":         tn / (tn + fn + eps),   # Valor predictivo negativo
        "F1":          2 * tp / (2 * tp + fp + fn + eps),
        "Accuracy":    (tp + tn) / (tp + tn + fp + fn + eps),
        "FPR":         fp / (fp + tn + eps),
        "FNR":         fn / (fn + tp + eps),
    }


def _interpretacion_binaria(metricas, clase_pos, clase_neg):
    """Devuelve lista de lineas con interpretacion para el caso binario."""
    L = []
    tp, tn, fp, fn = metricas["TP"], metricas["TN"], metricas["FP"], metricas["FN"]

    L.append("--- INTERPRETACION DETALLADA (binario) ---\n")
    L.append("Significado de cada celda:")
    L.append(f"  TP (Verdaderos Positivos)  = {tp}  -> casos clase '{clase_pos}' "
             f"que el modelo predijo correctamente")
    L.append(f"  TN (Verdaderos Negativos)  = {tn}  -> casos clase '{clase_neg}' "
             f"que el modelo predijo correctamente")
    L.append(f"  FP (Falsos Positivos)      = {fp}  -> casos clase '{clase_neg}' "
             f"que el modelo predijo como '{clase_pos}'  [error tipo I]")
    L.append(f"  FN (Falsos Negativos)      = {fn}  -> casos clase '{clase_pos}' "
             f"que el modelo predijo como '{clase_neg}'  [error tipo II]")
    L.append("")

    L.append("Metricas calculadas:")
    L.append(f"  Accuracy          = {metricas['Accuracy']:.3f}  "
             f"-> aciertos totales sobre el total de casos")
    L.append(f"  Precision         = {metricas['Precision']:.3f}  "
             f"-> de los predichos como '{clase_pos}', cuantos lo son realmente")
    L.append(f"  Recall (Sens.)    = {metricas['Recall']:.3f}  "
             f"-> de los '{clase_pos}' reales, cuantos capturamos")
    L.append(f"  Specificity       = {metricas['Specificity']:.3f}  "
             f"-> de los '{clase_neg}' reales, cuantos identificamos correctamente")
    L.append(f"  NPV               = {metricas['NPV']:.3f}  "
             f"-> de los predichos como '{clase_neg}', cuantos lo son realmente")
    L.append(f"  F1-score          = {metricas['F1']:.3f}  "
             f"-> balance entre Precision y Recall")
    L.append(f"  FPR (falsa alarma)= {metricas['FPR']:.3f}  "
             f"-> proporcion de '{clase_neg}' incorrectamente clasificados")
    L.append(f"  FNR (omision)     = {metricas['FNR']:.3f}  "
             f"-> proporcion de '{clase_pos}' que se nos escapan")
    L.append("")

    # Narrativa en lenguaje natural
    L.append("--- LECTURA EN LENGUAJE NATURAL ---")
    L.append(f"De cada 100 casos reales de '{clase_pos}', el modelo identifica "
             f"correctamente {metricas['Recall']*100:.0f}.")
    L.append(f"Cuando el modelo predice '{clase_pos}', acierta en "
             f"{metricas['Precision']*100:.0f} de cada 100 ocasiones.")
    L.append(f"De cada 100 casos reales de '{clase_neg}', el modelo los rechaza "
             f"correctamente en {metricas['Specificity']*100:.0f} ocasiones.")
    L.append("")

    # Recomendaciones automaticas
    L.append("--- RECOMENDACIONES ---")
    rec = []
    if metricas["Recall"] < 0.5:
        rec.append(f"Recall bajo ({metricas['Recall']:.2f}): el modelo deja escapar "
                   f"muchos '{clase_pos}'. Si capturarlos es prioridad, considera "
                   f"bajar el umbral de decision o reentrenar con mas datos positivos.")
    if metricas["Precision"] < 0.5:
        rec.append(f"Precision baja ({metricas['Precision']:.2f}): muchas veces que "
                   f"el modelo dice '{clase_pos}' se equivoca. Si los falsos positivos "
                   f"son costosos, sube el umbral o mejora las features.")
    if metricas["FPR"] > 0.3:
        rec.append(f"Tasa de falsa alarma alta ({metricas['FPR']:.2f}): el modelo "
                   f"marca como '{clase_pos}' a demasiados '{clase_neg}' reales.")
    if abs(metricas["Precision"] - metricas["Recall"]) > 0.2:
        rec.append("Hay un desequilibrio Precision-Recall importante. Revisa si las "
                   "clases estan balanceadas o si necesitas ajustar el umbral.")
    if metricas["Accuracy"] > 0.85 and metricas["F1"] < 0.5:
        rec.append("Accuracy alto pero F1 bajo: probable desbalance de clases. "
                   "Confia mas en F1 y Recall que en Accuracy.")
    if not rec:
        rec.append("Las metricas estan en niveles aceptables para todas las dimensiones.")
    for r in rec:
        L.append(f"  - {r}")
    L.append("")
    return L


def _interpretacion_multiclase(cm, cm_norm, clases, y_test, y_pred):
    """Devuelve interpretacion para el caso multiclase."""
    L = []
    L.append("--- INTERPRETACION DETALLADA (multiclase) ---\n")

    # Metricas por clase
    L.append("Desempeño por clase:")
    L.append(f"{'Clase':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Casos':>8}")
    for i, c in enumerate(clases):
        prec_c = cm[i, i] / (cm[:, i].sum() + 1e-9)
        rec_c = cm[i, i] / (cm[i, :].sum() + 1e-9)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-9)
        casos = int(cm[i, :].sum())
        L.append(f"{str(c):<15} {prec_c:>10.3f} {rec_c:>10.3f} {f1_c:>10.3f} {casos:>8d}")
    L.append("")

    # Mejor y peor clase
    recalls = {clases[i]: cm[i, i] / (cm[i, :].sum() + 1e-9) for i in range(len(clases))}
    mejor_clase = max(recalls, key=recalls.get)
    peor_clase = min(recalls, key=recalls.get)
    L.append(f"Clase mejor identificada: '{mejor_clase}' "
             f"(recall = {recalls[mejor_clase]:.3f})")
    L.append(f"Clase peor identificada:  '{peor_clase}' "
             f"(recall = {recalls[peor_clase]:.3f})")
    L.append("")

    # Pares de confusion (cuales se confunden mas)
    L.append("--- PRINCIPALES CONFUSIONES (errores entre clases) ---")
    pares = []
    for i in range(len(clases)):
        for j in range(len(clases)):
            if i != j and cm[i, j] > 0:
                pares.append((cm[i, j], clases[i], clases[j]))
    pares.sort(reverse=True)
    for cuenta, real, pred in pares[:5]:
        pct = 100 * cuenta / cm[clases == real if hasattr(clases, '__eq__')
                                else list(clases).index(real)].sum() if len(pares) else 0
        idx_real = list(clases).index(real)
        total_real = cm[idx_real, :].sum()
        pct = 100 * cuenta / total_real if total_real > 0 else 0
        L.append(f"  Real '{real}' -> Predicho '{pred}': {cuenta} casos "
                 f"({pct:.1f}% de los '{real}' reales)")
    L.append("")

    # Recomendaciones
    L.append("--- RECOMENDACIONES ---")
    rec = []
    if recalls[peor_clase] < 0.4:
        rec.append(f"La clase '{peor_clase}' es muy dificil de detectar "
                   f"(recall {recalls[peor_clase]:.2f}). Revisa si tiene pocas muestras "
                   f"o si necesita features mas informativas.")
    if pares and pares[0][0] > 0.3 * cm.sum() / len(clases):
        cuenta, real, pred = pares[0]
        rec.append(f"La confusion mas frecuente es '{real}' -> '{pred}'. "
                   f"Considera si estas clases son separables con los datos disponibles.")
    if not rec:
        rec.append("El modelo distingue razonablemente todas las clases.")
    for r in rec:
        L.append(f"  - {r}")
    L.append("")
    return L


def matriz_confusion(modelo_nombre, modelo, X_test, y_test, output_dir, reporte):
    """Matriz de confusion enriquecida con metricas e interpretacion completa."""
    reporte.append("=" * 70)
    reporte.append("6. MATRIZ DE CONFUSION E INTERPRETACION")
    reporte.append("=" * 70)
    reporte.append(f"Mejor modelo: {modelo_nombre}\n")

    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    clases = sorted(np.unique(y_test))

    # Matriz normalizada por fila (recall por clase visualizado)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    _generar_imagenes_matriz(cm, cm_norm, clases, modelo_nombre, output_dir)

    # Tabla de conteos
    reporte.append("Matriz de CONTEOS (filas=real, columnas=predicho):")
    reporte.append(pd.DataFrame(cm, index=[f"Real {c}" for c in clases],
                                 columns=[f"Pred {c}" for c in clases]).to_string())
    reporte.append("")

    # Tabla normalizada
    reporte.append("Matriz NORMALIZADA por fila (% de cada clase real):")
    reporte.append(pd.DataFrame(cm_norm.round(3),
                                 index=[f"Real {c}" for c in clases],
                                 columns=[f"Pred {c}" for c in clases]).to_string())
    reporte.append("")

    # Metricas globales avanzadas
    reporte.append("--- METRICAS GLOBALES ---")
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    try:
        mcc = matthews_corrcoef(y_test, y_pred)
    except Exception:
        mcc = float("nan")
    reporte.append(f"  Accuracy          = {(y_test == y_pred).mean():.3f}")
    reporte.append(f"  Balanced Accuracy = {bal_acc:.3f}  "
                   f"(accuracy ajustado por desbalance)")
    reporte.append(f"  Cohen's Kappa     = {kappa:.3f}  "
                   f"(acuerdo vs. azar; 1.0 perfecto, 0 = nivel azar)")
    reporte.append(f"  MCC               = {mcc:.3f}  "
                   f"(correlacion -1 a 1; robusto a desbalance)")
    reporte.append("")

    # Reporte sklearn estandar
    reporte.append("--- REPORTE DETALLADO POR CLASE (sklearn) ---")
    reporte.append(classification_report(y_test, y_pred, zero_division=0))

    # Interpretacion segun caso
    if len(clases) == 2:
        clase_neg, clase_pos = clases[0], clases[1]
        metricas = _metricas_binarias(cm)
        reporte.extend(_interpretacion_binaria(metricas, clase_pos, clase_neg))
    else:
        reporte.extend(_interpretacion_multiclase(cm, cm_norm,
                                                   np.array(clases), y_test, y_pred))


def resumen_ejecutivo(tabla, reporte):
    """Texto pensado para gerencia NO tecnica."""
    reporte.append("=" * 70)
    reporte.append("7. COMUNICACION A EQUIPO GERENCIAL NO TECNICO")
    reporte.append("=" * 70)

    test = tabla[tabla["conjunto"] == "Test"].copy()
    if len(test) == 0:
        return
    candidatos = test[~test["modelo"].str.lower().str.contains("baseline|dummy")]
    if len(candidatos) == 0:
        candidatos = test
    mejor = candidatos.sort_values("f1", ascending=False).iloc[0]

    auc_txt = f"{mejor['auc']:.2f}" if pd.notna(mejor["auc"]) else "no aplica"
    reporte.append(f"""
RESUMEN PARA GERENCIA:

Construimos varios modelos para apoyar la decision de a que clientes contactar.
El modelo que ofrece el mejor balance entre aciertos y errores es
'{mejor['modelo']}', con los siguientes resultados sobre datos que nunca vio:

  - De cada 100 decisiones, acierta aproximadamente {mejor['accuracy']*100:.0f}.
  - Tiene un puntaje balanceado (F1) de {mejor['f1']*100:.0f}% que combina
    no equivocarse al senalar clientes y no perder oportunidades.
  - La capacidad de distinguir entre clientes que responden y los que no
    (AUC) es {auc_txt} (0.5 seria decidir al azar; 1.0 seria perfecto).

QUE SIGNIFICA EN LA PRACTICA:
  - Podemos enfocar la campana en clientes con mayor probabilidad de
    responder, reduciendo el costo de contactos infructuosos.
  - Quedan errores: algunos clientes seran contactados sin responder
    (gasto extra) y otros no seran contactados aunque hubieran respondido
    (oportunidad perdida). La matriz de confusion cuantifica ambos.

RIESGOS Y RECOMENDACIONES:
  - Validar el modelo con datos mas recientes antes de pasarlo a produccion.
  - Establecer monitoreo: si la tasa de aciertos cae, reentrenar.
  - Acompanar la decision tecnica con criterio de negocio (clientes VIP,
    politicas de contacto, presupuesto).
""")
