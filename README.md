# Trabajo Unidad 2 - Minería de Datos

Pipeline completo en Python que ejecuta automáticamente todas las etapas:
1. EDA con limpieza automática (marcadores de nulo, comas decimales, moneda, %)
2. Partición Train/Validación/Test + Modelo Baseline (anti data leakage)
3. Clustering (K-means y Jerárquico) con índice Silhouette
4. Clasificación (Árbol de Decisión y Random Forest)
5. Evaluación comparativa (ROC, métricas, matriz de confusión)
6. Comunicación de resultados para audiencia no técnica

Funciona con **1 o 2 datasets** (si son dos, se comparan al final).

Hay **dos formas de usarlo**:
- App web interactiva con **Streamlit** (recomendada para visualizar)
- Script de consola con `main.py`

## Instalar dependencias

Abre la terminal en Visual Studio Code (`Ctrl + ñ`) y ejecuta:

```bash
pip install -r requirements.txt
```

## OPCIÓN A: App web con Streamlit (recomendada)

```bash
streamlit run app.py
```

Se abrirá automáticamente tu navegador en `http://localhost:8501`. En la barra lateral puedes:
- Elegir si vas a analizar 1 o 2 datasets
- Subir tu(s) CSV(s) arrastrando o seleccionando archivo
- (Opcional) escribir el nombre de la variable objetivo
- Pulsar **Ejecutar análisis**

La app muestra los resultados en pestañas:
1. EDA y Limpieza
2. Exploración
3. Partición y Baseline
4. Clustering (gráficos del codo, dendrograma, perfiles)
5. Clasificación
6. Evaluación (curvas ROC, tabla comparativa, mejor modelo destacado)
7. Matriz de Confusión
8. Resumen Gerencial

Cuando son 2 datasets, agrega una tercera pestaña externa con la comparación.
Puedes descargar el reporte completo en `.txt` desde la propia interfaz.

## OPCIÓN B: Script de consola

Modo interactivo (te pregunta todo):
```bash
python main.py
```

Modo por argumentos:
```bash
python main.py --data data/tu_archivo.csv --target columna_objetivo
python main.py --data data/a.csv --data2 data/b.csv
```

**Argumentos:**
- `--data` ruta al primer CSV
- `--data2` ruta al segundo CSV (opcional)
- `--target` / `--target2` columna objetivo (opcional, se autodetecta)
- `--output` carpeta de resultados (por defecto `output/`)

## Resultados

**Con 1 dataset** se generan en `output/`:
- `reporte_resultados.txt`
- `01_kmeans_codo_silhouette.png`
- `02_dendrograma.png`
- `03_perfiles_clusters.png`
- `04_curvas_roc.png`
- `05_matriz_confusion.png`
- `tabla_comparativa.csv`

**Con 2 datasets**: `output/dataset_1/`, `output/dataset_2/` y `output/comparacion/`.

## Estructura del proyecto

```
unidad2/
├── app.py                # App Streamlit
├── main.py               # Script de consola
├── requirements.txt
├── README.md
├── data/                 # CSVs de entrada (ejemplos incluidos)
├── src/
│   ├── data_loader.py    # Carga + detección de tipos
│   ├── eda.py            # EDA + limpieza automática
│   ├── partition.py      # Partición + baseline
│   ├── clustering.py     # K-means + jerárquico + silhouette
│   ├── classification.py # Árbol + Random Forest
│   ├── evaluation.py     # ROC + matriz confusión + resumen gerencial
│   └── comparar.py       # Comparación entre 2 datasets
└── output/               # Se genera al ejecutar
```
"# mineriadat" 
