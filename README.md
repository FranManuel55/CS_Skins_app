# CS2 · precio antes de salir

Proyecto integrador de **Ciencia de Datos** (UTN FRM, 2026) · grupo **5K10-13**: Paula Ferreyra, Francisco Manuel, Facundo Espinola, Agustín Behncke, Enzo Cervi, Ezequiel Buxo y Cristian Suarez.

**Pregunta:** ¿se puede estimar, sólo con los atributos que se conocen el día del lanzamiento, en qué percentil de precio va a quedar un skin de arma de Counter-Strike 2?

**Aplicación:** https://USUARIO.github.io/REPOSITORIO/ (GitHub Pages, carpeta `docs/`)

## Qué hay en el repositorio

| Carpeta / archivo | Qué es | Entrega |
| --- | --- | --- |
| `pipeline/cs2_pipeline.py` | DAG de Airflow (Astro) con 7 tareas: baja el catálogo de ByMykel y los precios de Steam, los limpia (bronce → plata), los une, valida y publica `cs2_skins_dataset_AAAA-MM-DD.csv` con su `calidad_AAAA-MM-DD.json` | 1 |
| `notebooks/EDA_percentil_del_dia.ipynb` | Análisis exploratorio: el percentil del día como objetivo, hipótesis H1–H4 | 2 |
| `notebooks/Entrega3_modelado_30-10_R2.ipynb` | Modelado: pisos, validación temporal por colección, comparación de cuatro familias con R², ajuste, prueba, error en dólares | 3 |
| `data/` | El snapshot del 30/10/2026 que usa la aplicación, su JSON de calidad, el embudo de filtros contado sobre el bronce y los números del notebook para verificar | — |
| `build_app.py` | Reproduce el modelo del notebook con la misma semilla y genera la aplicación | 4 |
| `app/plantilla.html` | Diseño de la página (HTML, CSS y JavaScript); `build_app.py` le inyecta los gráficos de Altair y los datos | 4 |
| `docs/index.html` | La aplicación generada, la que publica GitHub Pages | 4 |

## Cómo se corre todo

### 1 · El pipeline (Entrega 1)

Requiere Docker y la CLI de Astro.

```bash
astro dev init cs2-pipeline          # una sola vez
cp pipeline/cs2_pipeline.py cs2-pipeline/dags/
cd cs2-pipeline && astro dev start
```

En `http://localhost:8080`, disparar el DAG `cs2_pipeline` (parámetro `filas_objetivo`: 10000 para este snapshot). Las salidas quedan en `include/`: `cs2_skins_dataset_AAAA-MM-DD.csv` y `calidad_AAAA-MM-DD.json`. Copiarlas a `data/`.

### 2 · Los notebooks (Entregas 2 y 3)

Se abren en Colab o en Jupyter. El de la Entrega 3 fija `scikit-learn==1.6.1` en su primera celda; en Colab hay que ejecutarla, reiniciar la sesión y después *Ejecutar todo*. Busca el CSV en `include/` (local) o en la carpeta del Drive.

### 3 · La aplicación (Entrega 4)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_app.py                                            # snapshot del 30/10
python build_app.py --csv data/cs2_skins_dataset_AAAA-MM-DD.csv   # otro snapshot
```

Tarda alrededor de un minuto. El script:

1. lee el CSV del pipeline (y, si están al lado, su `calidad_*.json` y `embudo_*.json`);
2. reproduce las decisiones del notebook de la Entrega 3: la partición por colección, los cuatro pedazos temporales, los pisos, las cuatro familias comparadas con R², la grilla del elegido, la prueba, el error en dólares y la importancia por permutación;
3. calcula lo que dice el modelo de cada skin para *Explorá* (sin que el modelo haya visto su colección) y la predicción de cada combinación de atributos para *Fabricá un skin*;
4. arma los gráficos con Altair y escribe `docs/index.html`, una sola página con los datos adentro;
5. **comprueba que los números coinciden con el notebook** (`data/modelo_metadatos_notebook.json`) y corta si no coinciden. Con un snapshot de otra fecha avisa que los números son nuevos.

Si cambian los datos, se vuelve a correr y se hace commit de `docs/index.html`: GitHub Pages publica la versión nueva.

### Publicar en GitHub Pages

En el repositorio: *Settings → Pages → Build and deployment → Deploy from a branch*, rama `main`, carpeta `/docs`. La URL queda como `https://USUARIO.github.io/REPOSITORIO/`.

## El modelo, en una línea

`HistGradientBoostingRegressor` (`learning_rate=0.1`, `max_leaf_nodes=31`, `min_samples_leaf=40`, `loss="squared_error"`) dentro de un `Pipeline` con siete columnas: rareza, exterior, StatTrak, Souvenir, arma, origen y dificultad de drop. R² 0,614 ± 0,103 en validación temporal; 0,272 en las 16 colecciones más nuevas, donde no le gana a una tabla de rareza y exterior (0,308) porque las colecciones del Armory cambiaron lo que significa «drop de mapa».
