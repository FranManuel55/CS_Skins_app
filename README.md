# CS2 · precio antes de salir

Proyecto integrador de **Ciencia de Datos** (UTN FRM, 2026) · grupo **5K10-13**: Paula Ferreyra, Francisco Manuel, Facundo Espinola, Agustín Behncke, Enzo Cervi, Ezequiel Buxo y Cristian Suarez.

**Pregunta:** ¿se puede estimar, sólo con los atributos que se conocen el día del lanzamiento, en qué percentil de precio va a quedar un skin de arma de Counter-Strike 2?

**Aplicación:** https://franmanuel55.github.io/CS_Skins_app/ (GitHub Pages, carpeta `docs/`)

## Qué hay en el repositorio

| Carpeta / archivo | Qué es | Entrega |
| --- | --- | --- |
| `pipeline/cs2_pipeline.py` | DAG de Airflow (Astro) con 7 tareas: baja el catálogo de ByMykel y los precios de Steam, los limpia (bronce → plata), los une, valida y publica `cs2_skins_dataset_AAAA-MM-DD.csv` con su `calidad_AAAA-MM-DD.json` | 1 |
| `notebooks/EDA_percentil_del_dia.ipynb` | Análisis exploratorio: el percentil del día como objetivo, hipótesis H1–H4 | 2 |
| `notebooks/Entrega3_modelado_30-10_R2.ipynb` | Modelado: pisos, validación temporal por colección, comparación de cuatro familias con R², ajuste, prueba, error en dólares | 3 |
| `data/` | El snapshot del 30/10/2026 que usa la aplicación, su JSON de calidad, el embudo de filtros contado sobre el bronce y los números del notebook para verificar | — |
| `descargar_estaticos.py` | Baja los datos estáticos del juego desde el catálogo de ByMykel: rareza, colección, caja, rango de desgaste, el ícono de cada colección y cada arma sin diseño | 4 |
| `imagenes.py` | Baja una vez las fotos de cada skin desde Steam (tres por diseño, según el desgaste) a `docs/img/skins/` y escribe `data/imagenes_AAAA-MM-DD.json` | 4 |
| `data/estaticos/catalogo.json` | El resumen del catálogo que usa la página (lo escribe `descargar_estaticos.py`) | 4 |
| `build_app.py` | Reproduce el modelo del notebook con la misma semilla, corre el experimento del «drop nuevo» y genera la aplicación | 4 |
| `app/plantilla.html` | Diseño de la página (HTML, CSS y JavaScript); `build_app.py` le inyecta los gráficos de Altair y los datos | 4 |
| `docs/index.html` | La aplicación generada, la que publica GitHub Pages | 4 |
| `docs/img/` | Íconos de las colecciones y, si se bajaron, las miniaturas de los skins | 4 |

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
python descargar_estaticos.py               # catálogo, íconos y armas sin diseño (una vez)
python imagenes.py                          # fotos de los skins por desgaste (una vez; se puede cortar y retomar)
python build_app.py                                            # snapshot del 30/10
python build_app.py --csv data/cs2_skins_dataset_AAAA-MM-DD.csv   # otro snapshot
```

Las fotos quedan en `docs/img/` y la página usa esas copias. Si falta alguna, la pide a Steam; si tampoco carga, muestra el nombre del skin sobre el color de su rareza (y en «Probá el modelo», la silueta del arma).

Tarda alrededor de un minuto. El script:

1. lee el CSV del pipeline (y, si están al lado, su `calidad_*.json` y `embudo_*.json`);
2. reproduce las decisiones del notebook de la Entrega 3: la partición por colección, los cuatro pedazos temporales, los pisos, las cuatro familias comparadas con R², la grilla del elegido, la prueba, el error en dólares y la importancia por permutación;
3. calcula lo que dice el modelo de cada skin (sin que el modelo haya visto su colección) y la predicción de cada combinación de atributos para *Probá el modelo*;
4. repite el procedimiento con un corte en 03/2025 y el origen «drop nuevo» separado (la defensa de la sección *El modelo*);
5. arma los gráficos con Altair y escribe `docs/index.html`, una sola página con los datos adentro;
6. **comprueba que los números coinciden con el notebook** (`data/modelo_metadatos_notebook.json`) y corta si no coinciden. Con un snapshot de otra fecha avisa que los números son nuevos.

Si cambian los datos, se vuelve a correr y se hace commit de `docs/index.html`: GitHub Pages publica la versión nueva.

### Publicar en GitHub Pages

En el repositorio: *Settings → Pages → Build and deployment → Deploy from a branch*, rama `main`, carpeta `/docs`. La URL queda como `https://USUARIO.github.io/REPOSITORIO/`.

## El modelo, en una línea

`HistGradientBoostingRegressor` (`learning_rate=0.1`, `max_leaf_nodes=31`, `min_samples_leaf=40`, `loss="squared_error"`) dentro de un `Pipeline` con siete columnas: rareza, exterior, StatTrak, Souvenir, arma, origen y dificultad de drop. R² 0,614 ± 0,103 en validación temporal; 0,272 en las 16 colecciones más nuevas, donde no le gana a una tabla de rareza y exterior (0,308) porque las colecciones del Armory cambiaron lo que significa «drop de mapa».

## Cómo está contada la página

1. **La pregunta**: seis AK-47 reales, de USD 0,08 a USD 1.894.
2. **Primero, el juego** (para quien no juega): el nombre de un skin, la rareza, el exterior con una simulación del desgaste, StatTrak y Souvenir, cómo se consigue (con una caja que se puede abrir), el puesto en el mercado y un glosario plegable.
3. **El tallo (Inspeccionar)**: *Los datos*; *Lo que encontramos* (rareza y origen, en dos pasos); *El modelo* (A cómo lo probamos, B qué modelo, C skin por skin, D por qué se equivoca: el Armory, E cómo va a mejorar y qué nos llevamos). El Armory se cuenta donde lo descubrimos: en el modelo.
4. **La copa (Abrir la caja)**: *Explorá el mercado* (estante de colecciones por año, filtros, muro de skins y ficha con todas las versiones y lo que dijo el modelo) y *Probá el modelo* (el arma sin diseño, con el color de su rareza, StatTrak y el desgaste).

### La era Armory

Para el modelo, todo skin con colección que no sale de caja ni de paquete es «drop de mapa». La página separa, sólo para explicar, las colecciones de drop lanzadas desde la actualización *The Armory* (2/10/2024):

- **Armory** (canje con créditos de un pase): Graphic Design, Sport & Field, Overpass 2024, Train 2025, Spy Tech, Arabesque y las ediciones limitadas.
- **Paquete semanal**: Ascent, Boreal, Radiant (03/2025), Achroma, Harlequin (01/2026), y las terminales de Genesis (09/2025) y Dead Hand (03/2026).

Fuentes: notas de actualización de Valve reunidas en csgoskins.gg/updates, esports.net, tradeit.gg y csgodatabase.com/collection. El modelo entregado **no** usa esta división; `build_app.py` estima cuánto mejoraría si la usara (R² 0,32 → 0,81 en las colecciones desde 03/2025).

### Créditos de las imágenes

Imágenes de los skins y de las armas: Valve, a través del catálogo de ByMykel. El contador StatTrak™ y la cinta Souvenir están dibujados en la página. La simulación de desgaste es propia: se inspiró en cómo funciona un visor 3D profesional (Inspekt), sin usar su código ni las texturas del juego.
