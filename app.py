"""APP STREAMLIT - PIPELINE DE MINERIA DE DATOS (UNIDAD 2)"""

import os
import re
import sys
import tempfile
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import ejecutar_pipeline
from src.comparar import comparar_datasets

st.set_page_config(page_title="Mineria de Datos - Unidad 2",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
.hero-header { background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 50%, #3b82f6 100%);
    color: white; padding: 28px 32px; border-radius: 14px; margin-bottom: 22px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.18); }
.hero-header h1 { margin: 0; font-size: 28px; font-weight: 700; }
.hero-header p { margin: 6px 0 0 0; font-size: 14px; opacity: 0.9; }
.kpi { background: white; border-radius: 12px; padding: 16px 18px;
    border-left: 4px solid #3b82f6; box-shadow: 0 1px 3px rgba(0,0,0,0.06); height: 100%; }
.kpi-label { font-size: 12px; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.6px; font-weight: 600; }
.kpi-value { font-size: 26px; font-weight: 700; color: #0f172a; margin-top: 4px; }
.kpi-hint { font-size: 12px; color: #94a3b8; margin-top: 2px; }
.kpi-success { border-left-color: #10b981; }
.kpi-warning { border-left-color: #f59e0b; }
.kpi-danger  { border-left-color: #ef4444; }
.kpi-info    { border-left-color: #6366f1; }
.section-title { font-size: 20px; font-weight: 700; color: #0f172a;
    margin: 18px 0 10px 0; padding-bottom: 6px; border-bottom: 2px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #1e3a8a !important; border-bottom-color: #3b82f6 !important; }
.insight-box { background: #eff6ff; border-left: 4px solid #3b82f6;
    padding: 12px 16px; border-radius: 8px; margin: 10px 0; color: #1e3a8a; font-size: 14px; }
.insight-box-success { background: #ecfdf5; border-left-color: #10b981; color: #065f46; }
.insight-box-warning { background: #fffbeb; border-left-color: #f59e0b; color: #92400e; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""<div class="hero-header"><h1>Pipeline de Mineria de Datos &mdash; Unidad 2</h1>
<p>EDA + limpieza &middot; Discretizacion &middot; Particion &middot; Clustering &middot;
Clasificacion &middot; Matriz de confusion enriquecida</p></div>""", unsafe_allow_html=True)


def kpi_card(label, value, hint="", variant=""):
    cls = "kpi"
    if variant in ("success", "warning", "danger", "info"):
        cls = f"kpi kpi-{variant}"
    st.markdown(f'<div class="{cls}"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-hint">{hint}</div></div>', unsafe_allow_html=True)


def insight(texto, tipo="info"):
    cls = {"info": "insight-box", "success": "insight-box insight-box-success",
           "warning": "insight-box insight-box-warning"}.get(tipo, "insight-box")
    st.markdown(f'<div class="{cls}">{texto}</div>', unsafe_allow_html=True)


def section(titulo):
    st.markdown(f'<div class="section-title">{titulo}</div>', unsafe_allow_html=True)


def parsear_secciones(texto):
    secciones = {"eda": [], "exploracion": [], "particion": [], "clustering": [],
                 "clasificacion": [], "comparativa": [], "confusion": [], "gerencia": []}
    actual = None
    for linea in texto.split("\n"):
        u = linea.upper()
        if "EDA" in u and "LIMPIEZA" in u:
            actual = "eda"
        elif ("EXPLORACION" in u or "EXPLORACI" in u) and "INICIAL" in u:
            actual = "exploracion"
        elif "PARTICION" in u or "PARTICI" in u or "BASELINE" in u:
            actual = "particion"
        elif "SEGMENTACION" in u or "SEGMENTACI" in u or "CLUSTERING" in u:
            actual = "clustering"
        elif "CLASIFICACION" in u or "CLASIFICACI" in u:
            if "EVALUACION" not in u and "EVALUACI" not in u:
                actual = "clasificacion"
        elif "EVALUACION COMPARATIVA" in u or ("EVALUACI" in u and "COMPARATIVA" in u):
            actual = "comparativa"
        elif "MATRIZ DE CONFUSION" in u or "MATRIZ DE CONFUSI" in u:
            actual = "confusion"
        elif "COMUNICACION" in u or "COMUNICACI" in u or "GERENCIAL" in u:
            actual = "gerencia"
        if actual:
            secciones[actual].append(linea)
    return {k: "\n".join(v).strip() for k, v in secciones.items()}


def extraer_metricas_mejor(tabla):
    test = tabla[tabla["conjunto"].str.contains("est", case=False, na=False)]
    test = test[~test["modelo"].str.contains("aseline|ummy", case=False, na=False)]
    if len(test) == 0:
        return None
    mejor = test.sort_values("f1", ascending=False).iloc[0]
    return {"modelo": str(mejor["modelo"]),
            "accuracy": float(mejor["accuracy"]), "f1": float(mejor["f1"]),
            "auc": float(mejor["auc"]) if pd.notna(mejor["auc"]) else None}


# Helpers de parseo para la matriz
def extraer_seccion(texto, marca):
    """Extrae contenido entre '--- MARCA ...' y el siguiente '---' o '===='."""
    patron = rf'---\s+{re.escape(marca)}[^\n]*\n(.*?)(?=\n---\s+[A-Z]|\n====|\Z)'
    mm = re.search(patron, texto, re.DOTALL)
    return mm.group(1).strip() if mm else ""


def parsear_celdas_binarias(seccion):
    """Saca TP/TN/FP/FN del bloque de interpretacion binaria."""
    d = {}
    for codigo in ["TP", "TN", "FP", "FN"]:
        pat = rf"{codigo}\s*\([^)]+\)\s*=\s*(\d+)\s*->\s*(.*)"
        mm = re.search(pat, seccion)
        if mm:
            d[codigo] = {"valor": int(mm.group(1)), "desc": mm.group(2).strip()}
    return d


def parsear_classification_report(seccion):
    """Convierte el classification_report en DataFrame."""
    filas = []
    for l in seccion.split("\n"):
        m = re.match(r"^\s*(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", l)
        if m:
            nombre = m.group(1)
            if nombre in ("precision", "recall", "f1-score", "support"):
                continue
            filas.append({"Clase": nombre, "Precision": float(m.group(2)),
                          "Recall": float(m.group(3)), "F1": float(m.group(4)),
                          "Casos": int(m.group(5))})
    return pd.DataFrame(filas) if filas else None


# ---------------------------- SIDEBAR ----------------------------
with st.sidebar:
    st.markdown("### Configuracion")
    modo = st.radio("Cantidad de datasets",
                    options=["1 dataset", "2 datasets"], index=0)

    st.markdown("---")
    st.markdown("**Dataset 1**")
    archivo1 = st.file_uploader("CSV", type=["csv"], key="f1",
                                 label_visibility="collapsed")
    target1 = st.text_input("Variable objetivo (opcional)", key="t1",
                             placeholder="Dejar vacio para autodetectar")

    archivo2 = None
    target2 = None
    if modo == "2 datasets":
        st.markdown("---")
        st.markdown("**Dataset 2**")
        archivo2 = st.file_uploader("CSV", type=["csv"], key="f2",
                                     label_visibility="collapsed")
        target2 = st.text_input("Variable objetivo (opcional)", key="t2",
                                 placeholder="Dejar vacio para autodetectar")

    st.markdown("---")
    st.markdown("**Discretizacion del target**")
    st.caption("Solo se aplica si el target es continuo.")
    metodo_disc = st.selectbox("Metodo",
        options=["cuantiles", "uniforme", "binario_mediana", "binario_media"],
        format_func=lambda x: {"cuantiles": "Cuantiles (grupos balanceados)",
                                "uniforme": "Intervalos uniformes",
                                "binario_mediana": "Binario por mediana",
                                "binario_media": "Binario por media"}.get(x, x))
    n_bins_disc = st.slider("Numero de grupos", 2, 5, 3)

    st.markdown("---")
    ejecutar = st.button("Ejecutar analisis", type="primary",
                         use_container_width=True)


# ---------------------------- RENDER ----------------------------
def mostrar_resultados(output_dir, resultado=None):
    ruta = os.path.join(output_dir, "reporte_resultados.txt")
    if not os.path.exists(ruta):
        st.error("Reporte no encontrado.")
        return

    with open(ruta, encoding="utf-8") as f:
        texto = f.read()
    secciones = parsear_secciones(texto)

    # KPIs superiores
    if resultado:
        m = extraer_metricas_mejor(resultado["tabla"])
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Filas", f"{resultado['filas']:,}",
                     hint=f"{resultado['columnas']} columnas", variant="info")
        with c2:
            kpi_card("Variable objetivo", resultado["target"], variant="info")
        with c3:
            clust = resultado["clustering"]
            sil = clust.get("kmeans_silhouette")
            kpi_card("Silhouette K-Means",
                     f"{sil:.3f}" if sil is not None else "N/A",
                     hint=f"k = {clust.get('kmeans_k') or 'N/A'}",
                     variant="success" if (sil or 0) > 0.4 else "warning")
        with c4:
            if m:
                f1 = m["f1"]
                v = "success" if f1 > 0.75 else "warning" if f1 > 0.6 else "danger"
                kpi_card("F1 mejor modelo", f"{f1:.3f}",
                         hint=m["modelo"], variant=v)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.download_button("Descargar reporte completo (.txt)",
                       data=texto, file_name="reporte_resultados.txt",
                       mime="text/plain")

    tabs = st.tabs(["Resumen", "EDA", "Particion", "Clustering",
                    "Clasificacion", "Evaluacion",
                    "Matriz de Confusion", "Para Gerencia"])

    # TAB 0: Resumen
    with tabs[0]:
        section("Resumen del analisis")
        if resultado:
            m = extraer_metricas_mejor(resultado["tabla"])
            if m:
                col1, col2, col3 = st.columns(3)
                with col1:
                    kpi_card("Accuracy", f"{m['accuracy']*100:.1f}%",
                             hint="aciertos sobre el total", variant="success")
                with col2:
                    kpi_card("F1-score", f"{m['f1']:.3f}",
                             hint="balance precision/recall", variant="success")
                with col3:
                    auc = m["auc"]
                    kpi_card("AUC", f"{auc:.3f}" if auc else "N/A",
                             hint="capacidad discriminativa",
                             variant="success" if (auc or 0) > 0.8 else "info")
                insight(f"El modelo <b>{m['modelo']}</b> es el ganador. "
                        f"De cada 100 predicciones acierta {m['accuracy']*100:.0f}, "
                        f"con un F1 de {m['f1']:.2f}.", tipo="success")
        with st.expander("Ver tabla comparativa completa"):
            tabla_csv = os.path.join(output_dir, "tabla_comparativa.csv")
            if os.path.exists(tabla_csv):
                st.dataframe(pd.read_csv(tabla_csv),
                             use_container_width=True, hide_index=True)

    # TAB 1: EDA
    with tabs[1]:
        section("EDA con limpieza automatica")
        insight("Detecta marcadores no estandar de nulos, reconvierte strings "
                "numericos (comas, moneda, %) y elimina duplicados.", tipo="info")
        with st.expander("Ver detalle", expanded=True):
            st.code(secciones["eda"], language=None)

    # TAB 2: Particion
    with tabs[2]:
        section("Particion 60/20/20 y Baseline")
        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Entrenamiento", "60%", hint="ajusta modelos", variant="info")
        with c2: kpi_card("Validacion", "20%", hint="seleccion", variant="info")
        with c3: kpi_card("Prueba", "20%", hint="evaluacion final", variant="info")
        insight("<b>Anti data leakage:</b> imputacion, escalado y OneHotEncoder "
                "se ajustan SOLO con train.", tipo="warning")
        with st.expander("Ver detalle"):
            st.code(secciones["particion"], language=None)

    # TAB 3: Clustering
    with tabs[3]:
        section("Segmentacion K-Means y Jerarquico")
        if resultado:
            c = resultado["clustering"]
            col1, col2 = st.columns(2)
            with col1:
                kpi_card("K-Means", f"k = {c.get('kmeans_k') or 'N/A'}",
                         hint=f"Silhouette = {c.get('kmeans_silhouette') or 0:.3f}",
                         variant="success")
            with col2:
                kpi_card("Jerarquico", f"k = {c.get('jerarquico_k') or 'N/A'}",
                         hint=f"Silhouette = {c.get('jerarquico_silhouette') or 0:.3f}",
                         variant="success")
        col1, col2 = st.columns(2)
        img_codo = os.path.join(output_dir, "01_kmeans_codo_silhouette.png")
        img_dendro = os.path.join(output_dir, "02_dendrograma.png")
        if os.path.exists(img_codo):
            with col1:
                with st.container(border=True):
                    st.markdown("**Codo + Silhouette K-Means**")
                    st.image(img_codo, use_container_width=True)
        if os.path.exists(img_dendro):
            with col2:
                with st.container(border=True):
                    st.markdown("**Dendrograma**")
                    st.image(img_dendro, use_container_width=True)
        img_perfiles = os.path.join(output_dir, "03_perfiles_clusters.png")
        if os.path.exists(img_perfiles):
            with st.container(border=True):
                st.markdown("**Perfiles de los clusters**")
                st.image(img_perfiles, use_container_width=True)
        with st.expander("Ver detalle textual"):
            st.code(secciones["clustering"], language=None)

    # TAB 4: Clasificacion
    with tabs[4]:
        section("Arbol de Decision y Random Forest")
        if resultado:
            tabla = resultado["tabla"]
            test = tabla[tabla["conjunto"].str.contains("est", case=False, na=False)]
            test = test[~test["modelo"].str.contains("aseline|ummy", case=False, na=False)]
            if len(test) > 0:
                cols = st.columns(len(test))
                for col, (_, fila) in zip(cols, test.iterrows()):
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**{fila['modelo']}**")
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("Acc", f"{fila['accuracy']:.3f}")
                            mc2.metric("F1", f"{fila['f1']:.3f}")
                            mc3.metric("AUC",
                                       f"{fila['auc']:.3f}" if pd.notna(fila['auc']) else "N/A")
        with st.expander("Ver detalle textual"):
            st.code(secciones["clasificacion"], language=None)

    # TAB 5: Evaluacion
    with tabs[5]:
        section("Evaluacion comparativa")
        col1, col2 = st.columns([3, 2])
        img_roc = os.path.join(output_dir, "04_curvas_roc.png")
        with col1:
            with st.container(border=True):
                st.markdown("**Curvas ROC**")
                if os.path.exists(img_roc):
                    st.image(img_roc, use_container_width=True)
                else:
                    st.info("ROC no disponible (problema multiclase).")
        with col2:
            with st.container(border=True):
                st.markdown("**Tabla comparativa**")
                tabla_csv = os.path.join(output_dir, "tabla_comparativa.csv")
                if os.path.exists(tabla_csv):
                    st.dataframe(pd.read_csv(tabla_csv),
                                 use_container_width=True, hide_index=True,
                                 height=320)
        with st.expander("Ver detalle textual"):
            st.code(secciones["comparativa"], language=None)

    # TAB 6: Matriz de Confusion ENRIQUECIDA (NUEVA VERSION CON PARSER CORRECTO)
    with tabs[6]:
        section("Matriz de Confusion e interpretacion")

        img_cm = os.path.join(output_dir, "05_matriz_confusion.png")
        if os.path.exists(img_cm):
            with st.container(border=True):
                st.image(img_cm, use_container_width=True,
                         caption="Izquierda: conteos. Derecha: % por fila.")

        texto_conf = secciones["confusion"]
        sub_tabs = st.tabs(["Metricas globales", "Por clase",
                            "Interpretacion", "Recomendaciones"])

        # METRICAS GLOBALES
        with sub_tabs[0]:
            patrones_g = [
                (r"Accuracy\s+=\s+([0-9.]+)", "Accuracy", "aciertos / total"),
                (r"Balanced Accuracy\s+=\s+([0-9.]+)", "Balanced Acc.",
                 "ajustado por desbalance"),
                (r"Cohen's Kappa\s+=\s+(-?[0-9.]+)", "Cohen's Kappa",
                 "acuerdo vs. azar"),
                (r"MCC\s+=\s+(-?[0-9.]+)", "MCC", "robusto a desbalance"),
            ]
            cols = st.columns(4)
            for col, (pat, label, hint) in zip(cols, patrones_g):
                mm = re.search(pat, texto_conf)
                if mm:
                    val = float(mm.group(1))
                    if "Kappa" in label or "MCC" in label:
                        v = "success" if val > 0.6 else "warning" if val > 0.3 else "danger"
                    else:
                        v = "success" if val > 0.75 else "warning" if val > 0.5 else "danger"
                    with col:
                        kpi_card(label, f"{val:.3f}", hint=hint, variant=v)

            bin_m = ["Precision", "Recall", "Specificity", "NPV",
                     "F1-score", "FPR", "FNR"]
            valores = {}
            for nm in bin_m:
                pat = rf"{re.escape(nm)}[^=]*=\s*([0-9.]+)"
                mm = re.search(pat, texto_conf)
                if mm:
                    valores[nm] = float(mm.group(1))
            if len(valores) >= 4:
                st.markdown("&nbsp;")
                st.markdown("##### Metricas binarias adicionales")
                hints_b = {"Precision": "% predicciones positivas correctas",
                           "Recall": "% positivos reales capturados",
                           "Specificity": "% negativos reales capturados",
                           "NPV": "% predicciones negativas correctas",
                           "F1-score": "balance precision-recall",
                           "FPR": "falsa alarma (menor mejor)",
                           "FNR": "omision (menor mejor)"}
                items = list(valores.items())
                for i in range(0, len(items), 4):
                    cols_r = st.columns(4)
                    for col, (name, val) in zip(cols_r, items[i:i+4]):
                        if name in ("FPR", "FNR"):
                            v = "success" if val < 0.15 else "warning" if val < 0.3 else "danger"
                        else:
                            v = "success" if val > 0.75 else "warning" if val > 0.5 else "danger"
                        with col:
                            kpi_card(name, f"{val:.3f}",
                                     hint=hints_b.get(name, ""), variant=v)

        # POR CLASE - con dataframe coloreado
        with sub_tabs[1]:
            sec_cls = extraer_seccion(texto_conf, "REPORTE DETALLADO POR CLASE")
            df_cls = parsear_classification_report(sec_cls)
            if df_cls is not None and len(df_cls) > 0:
                with st.container(border=True):
                    st.markdown("##### Desempeno por clase")
                    styled = (df_cls.style
                              .background_gradient(subset=["Precision", "Recall", "F1"],
                                                    cmap="RdYlGn", vmin=0, vmax=1)
                              .format({"Precision": "{:.3f}", "Recall": "{:.3f}",
                                       "F1": "{:.3f}", "Casos": "{:d}"}))
                    st.dataframe(styled, use_container_width=True, hide_index=True)
                if len(df_cls) > 1:
                    st.markdown("&nbsp;")
                    mejor_c = df_cls.loc[df_cls["Recall"].idxmax()]
                    peor_c = df_cls.loc[df_cls["Recall"].idxmin()]
                    col1, col2 = st.columns(2)
                    with col1:
                        kpi_card("Clase mejor identificada", str(mejor_c["Clase"]),
                                 hint=f"Recall = {mejor_c['Recall']:.3f}",
                                 variant="success")
                    with col2:
                        kpi_card("Clase peor identificada", str(peor_c["Clase"]),
                                 hint=f"Recall = {peor_c['Recall']:.3f}",
                                 variant="danger")
            else:
                st.warning("No se pudo parsear el reporte por clase.")

        # INTERPRETACION - cards visuales en vez de bloque gris
        with sub_tabs[2]:
            sec_bin = extraer_seccion(texto_conf, "INTERPRETACION DETALLADA")
            celdas = parsear_celdas_binarias(sec_bin) if sec_bin else {}

            if celdas:
                with st.container(border=True):
                    st.markdown("##### Significado de cada celda de la matriz")
                    c1, c2 = st.columns(2)
                    if "TP" in celdas:
                        with c1:
                            kpi_card("TP - Verdaderos Positivos",
                                     str(celdas["TP"]["valor"]),
                                     hint=celdas["TP"]["desc"][:100],
                                     variant="success")
                    if "TN" in celdas:
                        with c2:
                            kpi_card("TN - Verdaderos Negativos",
                                     str(celdas["TN"]["valor"]),
                                     hint=celdas["TN"]["desc"][:100],
                                     variant="success")
                    c3, c4 = st.columns(2)
                    if "FP" in celdas:
                        with c3:
                            kpi_card("FP - Falsos Positivos (Error tipo I)",
                                     str(celdas["FP"]["valor"]),
                                     hint=celdas["FP"]["desc"][:100],
                                     variant="warning")
                    if "FN" in celdas:
                        with c4:
                            kpi_card("FN - Falsos Negativos (Error tipo II)",
                                     str(celdas["FN"]["valor"]),
                                     hint=celdas["FN"]["desc"][:100],
                                     variant="danger")

            sec_nat = extraer_seccion(texto_conf, "LECTURA EN LENGUAJE NATURAL")
            if sec_nat:
                st.markdown("&nbsp;")
                with st.container(border=True):
                    st.markdown("##### Lectura en lenguaje natural")
                    for frase in sec_nat.split("\n"):
                        f = frase.strip()
                        if f:
                            insight(f, tipo="info")

            sec_cf = extraer_seccion(texto_conf, "PRINCIPALES CONFUSIONES")
            if sec_cf:
                st.markdown("&nbsp;")
                with st.container(border=True):
                    st.markdown("##### Principales confusiones (errores entre clases)")
                    for linea in sec_cf.split("\n"):
                        l = linea.strip()
                        if l.startswith("-") or "->" in l:
                            insight(l.lstrip("- ").strip(), tipo="warning")

            if not celdas and not sec_nat and not sec_cf:
                st.info("No hay interpretacion adicional disponible.")

        # RECOMENDACIONES
        with sub_tabs[3]:
            sec_rec = extraer_seccion(texto_conf, "RECOMENDACIONES")
            if sec_rec:
                lineas_rec = []
                for l in sec_rec.split("\n"):
                    s = l.strip()
                    if s.startswith("-"):
                        lineas_rec.append(s.lstrip("- ").strip())
                if lineas_rec:
                    st.markdown("##### Acciones sugeridas")
                    for ln in lineas_rec:
                        positivo = any(p in ln.lower() for p in
                                       ["aceptable", "razonablemente", "bien", "buen"])
                        insight(ln, tipo="success" if positivo else "warning")
                else:
                    insight("Sin alertas: las metricas estan en niveles aceptables.",
                            tipo="success")
            else:
                insight("Sin recomendaciones disponibles.", tipo="info")

    # TAB 7: Gerencia
    with tabs[7]:
        section("Comunicacion a equipo gerencial NO tecnico")
        with st.container(border=True):
            txt = (secciones["gerencia"]
                   .replace("RESUMEN PARA GERENCIA:", "### Resumen para gerencia")
                   .replace("QUE SIGNIFICA EN LA PRACTICA:", "**Que significa en la practica:**")
                   .replace("RIESGOS Y RECOMENDACIONES:", "**Riesgos y recomendaciones:**"))
            st.markdown(txt)


# ---------------------------- EJECUTAR ----------------------------
if ejecutar:
    if archivo1 is None:
        st.error("Sube al menos el primer dataset.")
        st.stop()
    if modo == "2 datasets" and archivo2 is None:
        st.error("Seleccionaste 2 datasets pero solo subiste uno.")
        st.stop()

    tmp_dir = tempfile.mkdtemp(prefix="streamlit_unidad2_")
    ruta1 = os.path.join(tmp_dir, archivo1.name)
    with open(ruta1, "wb") as f:
        f.write(archivo1.getbuffer())

    ruta2 = None
    if archivo2 is not None:
        ruta2 = os.path.join(tmp_dir, archivo2.name)
        with open(ruta2, "wb") as f:
            f.write(archivo2.getbuffer())

    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    if modo == "1 dataset":
        with st.spinner("Ejecutando pipeline completo..."):
            try:
                res = ejecutar_pipeline(ruta1, target1 or None, output_dir,
                                        etiqueta=archivo1.name,
                                        metodo_discretizacion=metodo_disc,
                                        n_bins=n_bins_disc)
            except Exception as e:
                st.exception(e)
                st.stop()
        st.success("Analisis completado.")
        mostrar_resultados(output_dir, resultado=res)
    else:
        sub1 = os.path.join(output_dir, "dataset_1")
        sub2 = os.path.join(output_dir, "dataset_2")
        sub_comp = os.path.join(output_dir, "comparacion")
        try:
            with st.spinner(f"Procesando {archivo1.name}..."):
                res1 = ejecutar_pipeline(ruta1, target1 or None, sub1,
                                         etiqueta=archivo1.name,
                                         metodo_discretizacion=metodo_disc,
                                         n_bins=n_bins_disc)
            with st.spinner(f"Procesando {archivo2.name}..."):
                res2 = ejecutar_pipeline(ruta2, target2 or None, sub2,
                                         etiqueta=archivo2.name,
                                         metodo_discretizacion=metodo_disc,
                                         n_bins=n_bins_disc)
            with st.spinner("Generando comparacion..."):
                comparar_datasets(res1, res2, sub_comp)
        except Exception as e:
            st.exception(e)
            st.stop()

        st.success("Analisis completado para ambos datasets.")
        outer = st.tabs([f"Dataset 1: {archivo1.name}",
                         f"Dataset 2: {archivo2.name}",
                         "Comparacion final"])
        with outer[0]:
            mostrar_resultados(sub1, resultado=res1)
        with outer[1]:
            mostrar_resultados(sub2, resultado=res2)
        with outer[2]:
            section("Comparacion entre los dos datasets")
            m1 = extraer_metricas_mejor(res1["tabla"])
            m2 = extraer_metricas_mejor(res2["tabla"])
            if m1 and m2:
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.markdown(f"**{res1['nombre']}**")
                        a, b, c = st.columns(3)
                        a.metric("Acc", f"{m1['accuracy']:.3f}")
                        b.metric("F1", f"{m1['f1']:.3f}")
                        c.metric("AUC", f"{m1['auc']:.3f}" if m1['auc'] else "N/A")
                with col2:
                    with st.container(border=True):
                        st.markdown(f"**{res2['nombre']}**")
                        a, b, c = st.columns(3)
                        a.metric("Acc", f"{m2['accuracy']:.3f}",
                                 delta=f"{(m2['accuracy']-m1['accuracy'])*100:+.1f}pp")
                        b.metric("F1", f"{m2['f1']:.3f}",
                                 delta=f"{(m2['f1']-m1['f1']):+.3f}")
                        if m1['auc'] and m2['auc']:
                            c.metric("AUC", f"{m2['auc']:.3f}",
                                     delta=f"{(m2['auc']-m1['auc']):+.3f}")
                        else:
                            c.metric("AUC", "N/A")
            comp_img = os.path.join(sub_comp, "comparacion_metricas.png")
            if os.path.exists(comp_img):
                with st.container(border=True):
                    st.markdown("**Comparacion visual**")
                    st.image(comp_img, use_container_width=True)
            comp_csv = os.path.join(sub_comp, "comparacion_datasets.csv")
            if os.path.exists(comp_csv):
                with st.container(border=True):
                    st.markdown("**Metricas lado a lado**")
                    st.dataframe(pd.read_csv(comp_csv),
                                 use_container_width=True, hide_index=True)
            comp_txt = os.path.join(sub_comp, "comparacion_datasets.txt")
            if os.path.exists(comp_txt):
                with st.expander("Interpretacion textual"):
                    with open(comp_txt, encoding="utf-8") as f:
                        st.code(f.read(), language=None)
else:
    col1, col2, col3 = st.columns(3)
    with col1: kpi_card("Paso 1", "Sube tu CSV", "barra lateral", "info")
    with col2: kpi_card("Paso 2", "Elige objetivo", "o autodetectar", "info")
    with col3: kpi_card("Paso 3", "Ejecutar", "menos de 1 min", "info")
    st.markdown("&nbsp;")
    insight("Pipeline completo de mineria de datos para tu trabajo Unidad 2.",
            tipo="info")
