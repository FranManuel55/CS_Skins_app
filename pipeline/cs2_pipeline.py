"""
### Pipeline de skins de CS2 — Proyecto Integrador

**Grupo 5K10-13 · Ciencia de Datos · UTN FRM 2026**

Produce un dataset tidy con precio y atributos de skins de CS2, uniendo dos
fuentes por `market_hash_name`.

**Una fila es** un skin de arma de CS2 (arma + skin + exterior + variante
StatTrak/Souvenir) observado en un snapshot de precio del Steam Community
Market. La **clave primaria** es `market_hash_name`. El pipeline publica
`sell_price_usd`, el precio crudo. La **variable objetivo** del modelo es el
percentil del precio dentro del dia, y se calcula en el notebook.

#### Arquitectura (medallion)

- `bronze/snapshot=<fecha>/` : datos crudos comprimidos, tal como llegaron
- `silver/`                  : datos ya procesados y tipados
- `cs2_skins_dataset_<fecha>.csv` : el dataset publicado

#### Reproducibilidad

El bronce se particiona por **fecha de captura**, y una pagina ya bajada nunca
se vuelve a pedir. Entonces: *mismo snapshot en bronce -> mismo CSV, siempre*.
Correrlo otro dia produce otro snapshot, y eso no es un bug: es que el mercado
se movio. El bronce guarda la evidencia de ambos. Con el parametro `snapshot`
se puede reprocesar un bronce viejo **sin pedirle nada a Steam**.

#### Cuanto se baja: `filas_objetivo`

Steam pagina de a **10 items** (ignora el parametro `count`) y limita por IP.
Bajar los 12.260 items de arma completos son ~1.226 pedidos y mas de 5 horas.
Por eso se pide una cantidad de filas y el pipeline arma el plan:

- `filas_objetivo = 2000` -> reparte las paginas entre las seis categorias **en
  proporcion a su tamano**, y dentro de cada una las distribuye **a lo largo de
  todo el catalogo** (muestreo sistematico).
- `filas_objetivo = 0` -> mercado de armas completo, sin muestreo.

Esto ultimo no es un detalle: Steam ordena por nombre, asi que pedir las
primeras N paginas es pedir el principio del alfabeto. Cuando el dataset es una
muestra, el reporte de calidad lo declara y calcula la cobertura. Una muestra
declarada es defendible; una muestra presentada como el mercado entero, no.

#### Por que se filtra por categoria en la URL

`category_730_Type[]` pide una categoria de arma por vez. Sin ese filtro Steam
devuelve el market entero (35.449 items: cajas, agentes, stickers, cuchillos,
guantes) y solo ~35 % sobrevive al merge. Con el filtro, cada item que baja es
una fila util: mismo dataset en un tercio del tiempo.

#### De donde sale cada skin: coleccion, caja y probabilidad de drop

`skins_not_grouped.json` **no trae la coleccion**: verificado sobre los 21.407
items, el campo no existe en el esquema (ni `collections`, ni `crates`, ni
`collection`). Por eso T1 baja tambien `collections.json` y `crates.json`, y T3
cruza **por `skin_id`**, no por nombre: `contains[].id` de cada coleccion es
exactamente el `skin_id` del catalogo. Cobertura medida: **99,93 %** de las
armas, una sola coleccion por skin, asi que el CSV sigue siendo tidy.

Con eso T3 calcula la probabilidad de que un drop sea exactamente ese item:

    p_item = p_rareza(peldano) / n_skins_del_peldano * p_wear * p_stattrak

- `p_rareza`: las odds oficiales de caja (79,92 / 15,98 / 3,2 / 0,64 %)
  aplicadas **por peldano de la escalera de rarezas de esa coleccion**, no por
  nombre de rareza: una caja de armas arranca en Mil-Spec y una coleccion
  souvenir en Consumer Grade.
- `p_wear`: la fraccion del rango `[min_float, max_float]` del skin que cae en
  la banda de ese exterior. **Supuesto declarado**: el float se sortea uniforme
  dentro de ese rango. Quedo *parcialmente validado* (el orden se sostiene,
  falla en los extremos), asi que `p_wear` se publica pero no entra al modelo.
- `p_stattrak`: 0,1.

`p_item` queda en **nulo estructural** para lo que no sale de una caja o es
souvenir (~37 % de las filas lo tienen). No es un dato que falte: es un dato que
no existe, porque esos items no tienen tabla de probabilidades.

#### De donde sale cada item: la columna `origen`

Cada fila se clasifica segun su mecanica de obtencion:

- `caja`: sale al abrir una caja de armas.
- `paquete souvenir`: version souvenir que sale de un paquete de torneo.
- `drop de mapa`: skin de una coleccion de mapa, que cae como drop sin caja.
- `sin identificar`: variantes que el catalogo genera pero que no estan en
  ningun contenedor conocido (por ejemplo, "Souvenir" de skins de caja). El
  inner join con Steam deja afuera las que nadie vende.

Es la forma de sumar las souvenir al analisis sin inventar probabilidades: el
modelo aprende solo si el origen mueve el precio.

#### Que NO calcula este pipeline

La variable objetivo es el **percentil del precio dentro del dia** (0 a 100), y
**no se calcula aca a proposito**: el DAG publica el precio crudo de cada
snapshot. El percentil es una decision de modelado y se calcula en el notebook,
agrupando por `fecha_snapshot`. Para un skin nuevo, se mide contra el mercado
que ya existe ese dia.
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.models.param import Param

# ------------------------------------------------------------------ grupo
GRUPO = "5K10-13"
TZ = ZoneInfo("America/Argentina/Buenos_Aires")

# ------------------------------------------------------------------ rutas
CARPETA_INCLUDE = Path("/usr/local/airflow/include")
CARPETA_BRONZE = CARPETA_INCLUDE / "bronze"
CARPETA_SILVER = CARPETA_INCLUDE / "silver"

# ------------------------------------------------------------------ fuentes
URL_CATALOGO = (
    "https://raw.githubusercontent.com/ByMykel/CSGO-API/"
    "main/public/api/en/skins_not_grouped.json"
)
# La pertenencia a colecciones y cajas vive en otros dos endpoints del mismo
# repositorio: el catalogo de skins no la publica.
URL_COLECCIONES = (
    "https://raw.githubusercontent.com/ByMykel/CSGO-API/"
    "main/public/api/en/collections.json"
)
URL_CAJAS = (
    "https://raw.githubusercontent.com/ByMykel/CSGO-API/"
    "main/public/api/en/crates.json"
)
STEAM_SEARCH = "https://steamcommunity.com/market/search/render/"
APPID_CS2 = 730

# ------------------------------------------------------------------ parametros
PAUSA_ENTRE_REQUESTS_SEG = 15
ITEMS_POR_PAGINA_PEDIDOS = 100   # se lo pedimos por si Steam vuelve a respetarlo
MAX_REINTENTOS = 6
CATEGORIAS_ARMAS = {"Rifles", "Pistols", "SMGs", "Heavy"}

# Tags de categoria del market de Steam. Son la Decision de alcance 1 del caso
# (solo armas) aplicada en la propia URL: cuchillos y guantes no se piden.
CATEGORIAS_STEAM = {
    "tag_CSGO_Type_Pistol": "Pistolas",
    "tag_CSGO_Type_SMG": "SMGs",
    "tag_CSGO_Type_Rifle": "Rifles",
    "tag_CSGO_Type_SniperRifle": "Francotiradores",
    "tag_CSGO_Type_Shotgun": "Escopetas",
    "tag_CSGO_Type_Machinegun": "Ametralladoras",
}

# Vocabularios ordinales. El exterior y la rareza no son categorias sueltas:
# tienen orden, y el modelo de la Unidad 3 lo va a querer asi.
EXTERIORES = ["Factory New", "Minimal Wear", "Field-Tested",
              "Well-Worn", "Battle-Scarred"]
ORDEN_EXTERIOR = {n: i for i, n in enumerate(EXTERIORES)}

RAREZAS = ["Consumer Grade", "Industrial Grade", "Mil-Spec Grade",
           "Restricted", "Classified", "Covert", "Contraband"]
ORDEN_RAREZA = {n: i for i, n in enumerate(RAREZAS)}

# ------------------------------------------------------- probabilidad de drop
# Bandas de float por exterior. Son fijas y publicas.
BANDAS_FLOAT = {
    "Factory New": (0.00, 0.07),
    "Minimal Wear": (0.07, 0.15),
    "Field-Tested": (0.15, 0.38),
    "Well-Worn": (0.38, 0.45),
    "Battle-Scarred": (0.45, 1.00),
}

# Odds oficiales de una caja, por PELDANO de la escalera de rarezas de la
# coleccion. 0,26 % es el slot de item especial raro (cuchillos y guantes),
# fuera de nuestro alcance.
ESCALERA_ODDS = [0.7992, 0.1598, 0.0320, 0.0064]
P_ITEM_ESPECIAL = 0.0026
P_STATTRAK = 0.1

# La API NO publica si una caja salio del pool activo de drops: es un dato
# externo y curado. Fuente: listado publico de estado de drop de cajas
# (steamdb.com/skins/cs2/cases, columna "Drop status"), consultado el
# 2026-09-23. Revisar antes de cada entrega: el pool rota.
#
# OJO, y hay que poder decirlo en la defensa: ordenadas por fecha, estas 25 son
# exactamente las 25 mas viejas. El corte cae entre Spectrum 2 (2017-09-06) y
# Clutch Case (2017-12-20), sin ninguna excepcion. La variable es, en la
# practica, una funcion escalon de la antiguedad: no usarla como si fuera
# informacion independiente de `antiguedad_col_dias`.
CAJAS_DISCONTINUADAS = {
    "CS:GO Weapon Case", "eSports 2013 Case", "Operation Bravo Case",
    "CS:GO Weapon Case 2", "eSports 2013 Winter Case",
    "Winter Offensive Weapon Case", "CS:GO Weapon Case 3",
    "Operation Phoenix Weapon Case", "Huntsman Weapon Case",
    "Operation Breakout Weapon Case", "eSports 2014 Summer Case",
    "Operation Vanguard Weapon Case", "Chroma Case", "Chroma 2 Case",
    "Falchion Case", "Shadow Case", "Revolver Case",
    "Operation Wildfire Case", "Chroma 3 Case", "Gamma Case",
    "Gamma 2 Case", "Glove Case", "Spectrum Case",
    "Operation Hydra Case", "Spectrum 2 Case",
}

# Columnas que aporta el catalogo y que pueden no venir. Si alguna queda 100 %
# nula se descarta con un aviso, en vez de romper el criterio "sin columnas
# vacias" o de tirar abajo la corrida por una fuente que declaramos opcional.
# `collection` YA NO esta en esta lista: paso de ser una columna opcional que
# se descartaba si venia vacia a ser una columna del modelo. Si el join falla,
# la corrida tiene que caerse en rojo (ver el chequeo de cobertura en T5).
COLUMNAS_ENRIQUECIMIENTO = ["min_float", "max_float"]

NULOS_ESPERADOS = {
    "collection": ("12 armas del catalogo no figuran en ninguna coleccion "
                   "publicada. Es un faltante real, no estructural (0,07 %)"),
    "collection_release": ("faltante real: el catalogo no publica la fecha "
                           "de lanzamiento de algunas colecciones"),
    "antiguedad_col_dias": "faltante real: depende de collection_release",
    "caja": ("ESTRUCTURAL: 52 de las 94 colecciones son de mapa o souvenir y "
             "no salen de una caja"),
    "caja_fecha": "ESTRUCTURAL: idem caja",
    "caja_discontinuada": "ESTRUCTURAL: idem caja",
    "p_rareza": "ESTRUCTURAL: solo se define para items que salen de una caja",
    "p_item": ("ESTRUCTURAL: solo para items de caja que no son souvenir; los "
               "souvenir vienen de paquetes de torneo, con otra mecanica"),
    "neg_log_p_item": "ESTRUCTURAL: idem p_item",
    "min_float": "el catalogo no cubre el 100 % de los skins",
    "max_float": "idem min_float",
}

DESCRIPCION_UNIDAD = (
    "un skin de arma de CS2 (arma + skin + exterior + variante "
    "StatTrak/Souvenir), observado en un snapshot de precio del Steam "
    "Community Market"
)

HEADERS_STEAM = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
    "Referer": f"https://steamcommunity.com/market/search?appid={APPID_CS2}",
}


# ================================================================== helpers


def _snapshot(context) -> str:
    """Fecha de captura del snapshot. Es el ancla de reproducibilidad.

    Se calcula en un solo lugar: el parametro si vino (permite reprocesar un
    bronce viejo), si no la fecha de la corrida en hora argentina.
    """
    explicito = context["params"].get("snapshot")
    if explicito:
        return str(explicito)

    # Se usa el momento en que la corrida ARRANCO, no su logical_date. En una
    # corrida programada con @daily, logical_date es el INICIO del intervalo
    # (medianoche UTC del dia anterior), y pasado a hora argentina caia un dia
    # antes de la captura real: el pendiente de la Entrega 1. start_date es
    # unico por corrida, asi que todas las tareas -y los reintentos- comparten
    # la misma etiqueta aunque crucen la medianoche.
    dag_run = context["dag_run"]
    momento = (getattr(dag_run, "start_date", None)
               or getattr(dag_run, "run_after", None)
               or getattr(dag_run, "logical_date", None))
    if momento is None:
        return datetime.now(TZ).strftime("%Y-%m-%d")
    return momento.astimezone(TZ).strftime("%Y-%m-%d")


def _carpeta_bronze(snapshot: str) -> Path:
    """El bronce vive particionado por fecha: una corrida de otro dia no pisa."""
    return CARPETA_BRONZE / f"snapshot={snapshot}"


def _escribir_gz(destino: Path, texto: str) -> None:
    """El crudo se guarda comprimido: son cientos de JSON casi identicos.

    Escritura atomica: se escribe a un .tmp y se renombra. Si la tarea muere a
    mitad de camino queda un .tmp huerfano, nunca un .json.gz truncado que la
    proxima corrida confunda con una pagina valida.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write(texto)
    tmp.replace(destino)


def _leer_gz(origen: Path) -> str:
    with gzip.open(origen, "rt", encoding="utf-8") as f:
        return f.read()


def _gz_valido(ruta: Path) -> bool:
    """Un archivo del bronce sirve solo si descomprime y parsea completo.

    "Existe" no alcanza: un corte a mitad de escritura deja un gzip truncado
    que existe pero no se puede leer.
    """
    try:
        json.loads(_leer_gz(ruta))
        return True
    except (EOFError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _pedir_a_steam(params: dict) -> tuple[str, dict]:
    """GET a Steam con reintentos ante HTTP 429.

    Devuelve el cuerpo crudo Y el parseado: lo que se guarda en bronce es la
    respuesta tal como llego, interpretarla es tarea de la capa plata.
    """
    for intento in range(1, MAX_REINTENTOS + 1):
        r = requests.get(STEAM_SEARCH, params=params,
                         headers=HEADERS_STEAM, timeout=60)

        if r.status_code == 429:
            # El Retry-After lo manda el servidor que nos esta frenando:
            # cualquier espera que inventemos nosotros es una conjetura peor.
            retry_after = int(r.headers.get("Retry-After", 0) or 0)
            espera = retry_after or (30 * intento)
            print(f"  429 recibido (intento {intento}/{MAX_REINTENTOS}), "
                  f"esperando {espera}s")
            time.sleep(espera)
            continue

        r.raise_for_status()
        return r.text, r.json()

    raise RuntimeError(f"Steam sigue devolviendo 429 tras {MAX_REINTENTOS} intentos")


def _params_pagina(tag: str, start: int) -> dict:
    """Parametros de una pagina de una categoria de arma.

    `sort_column=name` + `sort_dir=asc` fija el orden: sin eso Steam ordena por
    popularidad, que cambia entre pedidos, y la paginacion deja de ser
    deterministica.
    """
    return {
        "appid": APPID_CS2,
        "norender": 1,
        "currency": 1,                       # 1 = USD. Fijarlo importa: si no,
        "count": ITEMS_POR_PAGINA_PEDIDOS,   # Steam infiere moneda por IP y dos
        "start": start,                      # integrantes producen datasets
        "sort_column": "name",               # distintos.
        "sort_dir": "asc",
        "category_730_Type[]": tag,
    }


def _plan_de_starts(total: int, pagesize: int,
                    max_paginas: int | None = None) -> list[int]:
    """Que offsets pedir para una categoria.

    Sin `max_paginas`: cobertura completa (0, pagesize, 2*pagesize, ...).

    Con `max_paginas`: muestreo **sistematico**, repartiendo las paginas a lo
    largo de todo el catalogo en vez de tomar las primeras N. Steam ordena por
    nombre, asi que un prefijo del listado es un prefijo del alfabeto: pidiendo
    las 3 primeras paginas de cada categoria el dataset queda con una sola arma
    por categoria, y `weapon` deja de servir como variable del modelo.
    """
    if total <= 0:
        return []

    paginas_completas = -(-total // pagesize)          # techo
    if not max_paginas or max_paginas >= paginas_completas:
        return [i * pagesize for i in range(paginas_completas)]

    paso = total / max_paginas
    return [int(i * paso) for i in range(max_paginas)]


def _a_fecha_iso(valor) -> str | None:
    """Normaliza una fecha del catalogo a 'YYYY-MM-DD'.

    `crates.json` mezcla CINCO formatos: 2015-11-30, 2014/06/11, 2014/2/19,
    2014/11/5 y 2015/1/8. `pd.to_datetime` sobre la columna cruda levanta
    ValueError. Normalizar aca es literalmente para lo que existe la capa
    plata: el bronce guarda lo que llego, la plata lo deja usable.
    """
    if not valor:
        return None
    texto = str(valor).strip().replace("/", "-")
    partes = texto.split("-")
    if len(partes) == 3 and all(p.isdigit() for p in partes):
        try:
            a, m, d = (int(p) for p in partes)
            return datetime(a, m, d).strftime("%Y-%m-%d")
        except ValueError:
            pass
    print(f"AVISO: fecha no reconocida en el catalogo: {valor!r}")
    return None


def _bajar_json_a_bronce(url: str, destino: Path, rebajar: bool) -> Path:
    """Descarga un JSON del catalogo y lo guarda comprimido en el bronce."""
    if destino.exists() and not rebajar:
        if _gz_valido(destino):
            print(f"Ya en bronce: {destino.name}")
            return destino
        print(f"{destino.name} corrupto en bronce, se vuelve a descargar")

    respuesta = requests.get(url, timeout=120)
    respuesta.raise_for_status()
    _escribir_gz(destino, respuesta.text)
    print(f"Descargado: {destino.name} "
          f"({len(respuesta.content) / 1e6:.1f} MB crudos)")
    return destino


def _mapa_colecciones(colecciones: list) -> tuple[dict, dict, dict]:
    """De collections.json a tres diccionarios.

    El join es por `skin_id`, NO por nombre: `contains[].id` de una coleccion
    es exactamente el `skin_id` del catalogo de skins. Unir por nombre base
    obligaria a parsear "AK-47 | Redline (Field-Tested)", que se rompe con
    StatTrak, con Souvenir y con los nombres que llevan parentesis.
    """
    coleccion_de, salida_de, cajas_de = {}, {}, {}
    for col in colecciones:
        nombre = col.get("name")
        salida_de[nombre] = _a_fecha_iso(col.get("release_date"))
        cajas_de[nombre] = [c.get("name") for c in (col.get("crates") or [])]
        for item in col.get("contains") or []:
            coleccion_de[item.get("id")] = nombre
    return coleccion_de, salida_de, cajas_de


def _mapa_cajas(crates: list) -> dict:
    """Solo los contenedores de tipo Case: nombre -> fecha de primera venta.

    crates.json trae 481 contenedores de 10 tipos (capsulas de stickers,
    paquetes souvenir, music kits). Los 42 de tipo `Case` son los unicos con
    la tabla de probabilidades publicada que usamos.
    """
    return {c.get("name"): _a_fecha_iso(c.get("first_sale_date"))
            for c in crates if c.get("type") == "Case"}


def _validar_lista_curada(fecha_de_caja: dict) -> None:
    """Chequea que cada caja de la lista curada exista en el catalogo.

    Sin esto, un nombre mal escrito marca `caja_discontinuada = False` en
    silencio y nadie se entera. AVISA pero NO corta la corrida: desde el
    replanteo, `caja_discontinuada` ya no entra al modelo (es la antiguedad
    disfrazada), y no tiene sentido tirar abajo el DAG entero por una columna
    que solo se conserva como informacion.
    """
    fantasmas = sorted(CAJAS_DISCONTINUADAS - set(fecha_de_caja))
    if fantasmas:
        print("AVISO: CAJAS_DISCONTINUADAS tiene nombres que no existen en "
              f"crates.json: {fantasmas}. Revisar la constante; la corrida sigue.")
    else:
        print(f"Lista curada validada: {len(CAJAS_DISCONTINUADAS)} de "
              f"{len(fecha_de_caja)} cajas marcadas como discontinuadas")


def _skins_en_paquetes_souvenir(crates: list) -> set:
    """Los skin_id que aparecen en algun paquete souvenir.

    Se usa para distinguir una souvenir real -que sale de un paquete de
    torneo- de una variante que el catalogo genera pero que no tiene paquete.
    """
    ids = set()
    for c in crates:
        if c.get("type") in ("Souvenir", "Souvenir Highlight"):
            for clave in ("contains", "contains_rare"):
                for item in c.get(clave) or []:
                    ids.add(item.get("id"))
    return ids


def _origen(souvenir, skin_id, caja, coleccion, en_paquete_souvenir) -> str:
    """De donde sale el item, sin inventar probabilidades."""
    if souvenir and skin_id in en_paquete_souvenir:
        return "paquete souvenir"
    if not souvenir and caja:
        return "caja"
    if not souvenir and coleccion:
        return "drop de mapa"
    return "sin identificar"


def _probabilidad_de_exterior(min_float, max_float, exterior) -> float | None:
    """Que fraccion del rango de float del skin cae en la banda del exterior.

    SUPUESTO DECLARADO: el float de un drop se sortea uniforme dentro de
    [min_float, max_float]. Estado: parcialmente validado en el EDA de la
    Entrega 2 (el orden se sostiene, falla en los extremos), por eso p_wear se
    publica pero no entra al modelo.
    """
    banda = BANDAS_FLOAT.get(exterior)
    if banda is None or pd.isna(min_float) or pd.isna(max_float):
        return None
    rango = float(max_float) - float(min_float)
    if rango <= 0:
        return 1.0
    solape = min(banda[1], float(max_float)) - max(banda[0], float(min_float))
    return max(0.0, solape) / rango


def _escalera_de_odds(base: pd.DataFrame) -> tuple[dict, dict]:
    """Por coleccion, la probabilidad de cada rareza segun su peldano.

    Se calcula sobre el CATALOGO COMPLETO, no sobre la muestra descargada, asi
    que el denominador (cuantos skins comparten el peldano) es correcto aunque
    el dataset final sea una muestra.
    """
    odds, cuantos = {}, {}
    for coleccion, grupo in base.groupby("collection"):
        presentes = sorted(
            grupo["rarity"].dropna().unique(),
            key=lambda r: ORDEN_RAREZA.get(r, 99),
        )
        for peldano, rareza in enumerate(presentes):
            odds[(coleccion, rareza)] = (ESCALERA_ODDS[peldano]
                                         if peldano < len(ESCALERA_ODDS)
                                         else P_ITEM_ESPECIAL)
            cuantos[(coleccion, rareza)] = int(
                grupo.loc[grupo["rarity"] == rareza, "skin_id"].nunique())
    return odds, cuantos


def _buscar_snapshot_previo(snapshot_actual: str) -> Path | None:
    """Busca el CSV mas reciente distinto al del snapshot actual."""
    csvs = sorted(CARPETA_INCLUDE.glob("cs2_skins_dataset_*.csv"))
    csvs = [p for p in csvs
            if "_excel" not in p.name and snapshot_actual not in p.name]
    return csvs[-1] if csvs else None


# ================================================================== DAG


@dag(
    dag_id="cs2_pipeline",
    description="Pipeline de skins de CS2: atributos ByMykel + precios Steam",
    doc_md=__doc__,
    # Un snapshot de precios por dia. El bronce particionado por fecha hace que
    # cada corrida escriba su propia particion sin pisar la anterior.
    schedule="@daily",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    # Una corrida por vez: dos corridas simultaneas le piden a Steam a la vez
    # (y Steam las bloquea) y escriben en la misma particion del bronce.
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["cs2", "proyecto-integrador", GRUPO],
    params={
        "filas_objetivo": Param(
            2000, type="integer", minimum=0,
            title="Filas objetivo",
            description=(
                "Cuantas filas querer en el dataset. 0 = mercado de armas "
                "completo (~12.260 items, mas de 5 h). 2000 = ~200 paginas, "
                "~50 min. Las paginas se reparten entre las seis categorias en "
                "proporcion a su tamano y se distribuyen a lo largo del "
                "catalogo (muestreo sistematico), no las primeras N."
            ),
        ),
        "snapshot": Param(
            None, type=["null", "string"],
            title="Snapshot (YYYY-MM-DD)",
            description=(
                "Fecha de captura. Vacio = la fecha de la corrida. Fijarlo "
                "permite reprocesar un bronce ya bajado sin pedirle nada a "
                "Steam."
            ),
        ),
        "pausa_segundos": Param(
            float(PAUSA_ENTRE_REQUESTS_SEG), type="number",
            title="Espera entre pedidos",
            description=(
                "Segundos entre pagina y pagina. Por debajo de 15 este "
                "endpoint devuelve 429 con bastante seguridad."
            ),
        ),
        "rebajar_bronce": Param(
            False, type="boolean",
            title="Volver a descargar el bronce",
            description=(
                "Ignora las paginas ya guardadas y se las vuelve a pedir a "
                "Steam. Solo si sospechas que el crudo esta corrupto."
            ),
        ),
    },
)
def cs2_pipeline():

    # ---------------------------------------------------------- TAREA 1
    @task
    def descargar_catalogo(**context) -> dict:
        """**Bronce, fuente 2**: catalogo de ByMykel, en tres archivos.

        `skins_not_grouped.json` aporta los atributos que Steam no publica
        (rango de float, patron, rareza). `collections.json` y `crates.json`
        aportan **de donde sale cada skin**: el catalogo de skins no lo trae,
        verificado sobre los 21.407 items.

        Los tres van comprimidos a la particion del snapshot, sin interpretar.
        """
        snapshot = _snapshot(context)
        carpeta = _carpeta_bronze(snapshot)
        rebajar = context["params"]["rebajar_bronce"]

        rutas = {
            "skins": str(_bajar_json_a_bronce(
                URL_CATALOGO, carpeta / "catalogo_skins.json.gz", rebajar)),
            "colecciones": str(_bajar_json_a_bronce(
                URL_COLECCIONES, carpeta / "catalogo_colecciones.json.gz",
                rebajar)),
            "cajas": str(_bajar_json_a_bronce(
                URL_CAJAS, carpeta / "catalogo_cajas.json.gz", rebajar)),
        }
        return rutas

    # ---------------------------------------------------------- TAREA 2
    @task
    def descargar_precios_steam(**context) -> dict:
        """**Bronce, fuente 1**: precios del Steam Community Market.

        Pide una categoria de arma por vez (`category_730_Type[]`), asi cada
        item que baja es una fila util. Una pagina ya bajada no se vuelve a
        pedir nunca: es lo que hace instantanea la segunda corrida del dia y lo
        que vuelve viable un endpoint que pagina de a 10 y limita por IP.
        """
        snapshot = _snapshot(context)
        objetivo = int(context["params"]["filas_objetivo"] or 0)
        pausa = float(context["params"]["pausa_segundos"])
        rebajar = context["params"]["rebajar_bronce"]

        carpeta = _carpeta_bronze(snapshot) / "paginas_precios"

        # --- paso 1: cuantos items tiene cada categoria -------------------
        # El total_count viaja dentro de la propia pagina 0, asi que si ya esta
        # en bronce se lee de disco y no se le pide nada a Steam.
        categorias = []
        for tag, nombre in CATEGORIAS_STEAM.items():
            archivo0 = carpeta / f"{tag}" / "pagina_000000.json.gz"
            datos = None

            if archivo0.exists() and not rebajar:
                try:
                    datos = json.loads(_leer_gz(archivo0))
                    print(f"{nombre}: total_count leido del bronce")
                except Exception as e:
                    print(f"{nombre}: bronce ilegible ({e}), se re-pide")
                    datos = None

            if datos is None:
                cuerpo, datos = _pedir_a_steam(_params_pagina(tag, 0))
                _escribir_gz(archivo0, cuerpo)   # guardarla ahora evita pedirla
                time.sleep(pausa)                # de nuevo mas abajo

            categorias.append({
                "tag": tag,
                "nombre": nombre,
                "total": int(datos.get("total_count") or 0),
                "pagesize": int(datos.get("pagesize") or 10),
            })

        mercado = sum(c["total"] for c in categorias)
        if mercado <= 0:
            raise RuntimeError(
                "Ninguna categoria devolvio items. Puede haber cambiado el "
                "endpoint o tu IP estar limitada. Probar la URL en el navegador."
            )
        print(f"\nItems de arma en el market: {mercado}")

        # --- paso 2: repartir las paginas entre categorias ----------------
        # En proporcion al tamano de cada una. Darle N paginas fijas a todas
        # sobre-representa a las chicas: Escopetas tiene una fraccion de los
        # items de Pistolas y quedaria con el mismo peso en el dataset.
        for c in categorias:
            pagesize = c["pagesize"]
            if objetivo:
                cuota = c["total"] / mercado
                max_paginas = max(1, round(objetivo * cuota / pagesize))
            else:
                max_paginas = None
            c["starts"] = _plan_de_starts(c["total"], pagesize, max_paginas)

            cobertura = (len(c["starts"]) * pagesize / c["total"] * 100
                         if c["total"] else 0)
            print(f"  {c['nombre']:16s} {c['total']:6d} items -> "
                  f"{len(c['starts']):4d} paginas ({cobertura:.0f}% de la "
                  f"categoria)")

        planificados = sum(min(c["total"], len(c["starts"]) * c["pagesize"])
                           for c in categorias)
        total_paginas = sum(len(c["starts"]) for c in categorias)
        print(f"\nPLAN: {total_paginas} paginas, ~{planificados} items, "
              f"~{total_paginas * pausa / 60:.0f} min a {pausa}s por pedido")

        # --- paso 3: bajar -------------------------------------------------
        pedidas = reusadas = 0
        for c in categorias:
            for i, start in enumerate(c["starts"]):
                archivo = carpeta / c["tag"] / f"pagina_{start:06d}.json.gz"

                if archivo.exists() and not rebajar:
                    if _gz_valido(archivo):
                        reusadas += 1
                        continue
                    print(f"  {c['nombre']}: {archivo.name} corrupto en "
                          "bronce, se vuelve a pedir")

                cuerpo, datos = _pedir_a_steam(_params_pagina(c["tag"], start))
                if not (datos.get("results") or []):
                    print(f"  {c['nombre']}: pagina vacia en start={start}, "
                          "fin de categoria")
                    break

                _escribir_gz(archivo, cuerpo)
                pedidas += 1
                time.sleep(pausa)

            print(f"  {c['nombre']}: listo ({len(c['starts'])} paginas)")

        print(f"\nDescarga finalizada: {pedidas} paginas nuevas, "
              f"{reusadas} reusadas del bronce")

        return {
            "carpeta": str(carpeta),
            "snapshot": snapshot,
            "declarados": mercado,          # lo que Steam dice que hay
            "planificados": planificados,   # lo que nos propusimos traer
            "es_muestra": bool(objetivo and planificados < mercado),
        }

    # ---------------------------------------------------------- TAREA 3
    @task
    def procesar_atributos(rutas_catalogo: dict) -> str:
        """**Plata**: el JSON crudo de ByMykel a un CSV tidy de atributos.

        Aplica la Decision de alcance 1 del caso (solo armas), agrega las
        versiones ordinales de exterior y rareza, y resuelve **de donde sale
        cada skin**: coleccion, caja, y la probabilidad de que un drop sea
        exactamente ese item.
        """
        CARPETA_SILVER.mkdir(parents=True, exist_ok=True)

        skins = json.loads(_leer_gz(Path(rutas_catalogo["skins"])))
        colecciones = json.loads(_leer_gz(Path(rutas_catalogo["colecciones"])))
        cajas = json.loads(_leer_gz(Path(rutas_catalogo["cajas"])))
        print(f"Catalogo: {len(skins)} skins, {len(colecciones)} colecciones, "
              f"{len(cajas)} contenedores")

        coleccion_de, salida_de, cajas_de = _mapa_colecciones(colecciones)
        fecha_de_caja = _mapa_cajas(cajas)
        _validar_lista_curada(fecha_de_caja)
        en_paquete_souvenir = _skins_en_paquetes_souvenir(cajas)

        # Aplanamos item por item para manejar campos ausentes con .get() sin
        # depender de que todos los skins tengan la misma estructura.
        registros = []
        for item in skins:
            skin_id = item.get("skin_id")
            coleccion = coleccion_de.get(skin_id)
            # De los contenedores de la coleccion, el primero que sea una Case:
            # es el unico con tabla de probabilidades publicada.
            propias = [c for c in cajas_de.get(coleccion, [])
                       if c in fecha_de_caja]
            caja = propias[0] if propias else None

            registros.append({
                "market_hash_name": item.get("market_hash_name"),
                # skin_id NO es una feature: es la columna de agrupamiento. El
                # mismo skin base genera hasta 15 filas (exteriores x StatTrak
                # x Souvenir) y sin esto no se puede validar por grupo en la
                # Entrega 3: un split aleatorio dejaria filas hermanas de los
                # dos lados e inflaria las metricas.
                "skin_id": skin_id,
                "weapon": (item.get("weapon") or {}).get("name"),
                "category_name": (item.get("category") or {}).get("name"),
                "pattern": (item.get("pattern") or {}).get("name"),
                "rarity": (item.get("rarity") or {}).get("name"),
                "wear": (item.get("wear") or {}).get("name"),
                "stattrak": item.get("stattrak"),
                "souvenir": item.get("souvenir"),
                "min_float": item.get("min_float"),
                "max_float": item.get("max_float"),
                # --- origen del item ---
                "collection": coleccion,
                "collection_release": salida_de.get(coleccion),
                "caja": caja,
                "caja_fecha": fecha_de_caja.get(caja),
                "caja_discontinuada": ((caja in CAJAS_DISCONTINUADAS)
                                       if caja else None),
                "origen": _origen(bool(item.get("souvenir")), skin_id, caja,
                                  coleccion, en_paquete_souvenir),
            })

        df = pd.DataFrame(registros)

        df = df[df["category_name"].isin(CATEGORIAS_ARMAS)].copy()
        print(f"Items despues de filtrar armas: {len(df)}")

        antes = len(df)
        df = df.dropna(subset=["market_hash_name"])
        if antes != len(df):
            print(f"Filas sin market_hash_name eliminadas: {antes - len(df)}")

        antes = len(df)
        df = df.drop_duplicates(subset="market_hash_name", keep="first")
        print(f"Duplicados por market_hash_name eliminados: {antes - len(df)}")

        # Ordinales: el exterior y la rareza tienen orden natural y el modelo
        # de la Unidad 3 lo va a querer como numero, no como categoria suelta.
        df["wear_orden"] = df["wear"].map(ORDEN_EXTERIOR)
        df["rarity_orden"] = df["rarity"].map(ORDEN_RAREZA)

        # --- probabilidad de drop -----------------------------------------
        base = df.drop_duplicates(subset="skin_id")
        odds, cuantos = _escalera_de_odds(base)
        claves = list(zip(df["collection"], df["rarity"]))
        df["p_rareza"] = [odds.get(k) for k in claves]
        df["n_skins_rareza_col"] = [cuantos.get(k) for k in claves]
        df["p_wear"] = [
            _probabilidad_de_exterior(mn, mx, w)
            for mn, mx, w in zip(df["min_float"], df["max_float"], df["wear"])
        ]

        # p_item solo tiene sentido para lo que sale de una caja y no es
        # souvenir: los souvenir vienen de paquetes de torneo, con otra
        # mecanica. En el resto el nulo es ESTRUCTURAL y asi se declara en el
        # JSON: no es un dato que falte, es un dato que no existe.
        sale_de_caja = df["caja"].notna() & (df["souvenir"] != True)  # noqa: E712
        df["p_item"] = (
            df["p_rareza"] / df["n_skins_rareza_col"] * df["p_wear"]
            * df["stattrak"].map({True: P_STATTRAK, False: 1 - P_STATTRAK})
        ).where(sale_de_caja)
        df["neg_log_p_item"] = df["p_item"].map(
            lambda v: -math.log10(v) if pd.notna(v) and v > 0 else None)

        print("Origen de las filas:",
              df["origen"].value_counts(normalize=True).mul(100).round(1).to_dict())
        print(f"Cobertura: collection {df['collection'].notna().mean():.1%} | "
              f"caja {df['caja'].notna().mean():.1%} | "
              f"p_item {df['p_item'].notna().mean():.1%} | "
              f"colecciones distintas: {df['collection'].nunique()}")
        print(f"Filas por skin base: mediana "
              f"{df.groupby('skin_id').size().median():.0f}, "
              f"maximo {df.groupby('skin_id').size().max()}")

        ruta_salida = CARPETA_SILVER / "atributos.csv"
        df.to_csv(ruta_salida, index=False, encoding="utf-8")
        print(f"Atributos guardados en: {ruta_salida}")

        return str(ruta_salida)

    # ---------------------------------------------------------- TAREA 4
    @task
    def procesar_precios(info_descarga: dict) -> dict:
        """**Plata**: las paginas crudas del market a un CSV tidy de precios.

        Devuelve ademas cuantos items crudos trajo el bronce, que es el numero
        contra el que `unir_y_validar` mide si la descarga se corto a la mitad.
        """
        CARPETA_SILVER.mkdir(parents=True, exist_ok=True)

        archivos = sorted(Path(info_descarga["carpeta"]).rglob("pagina_*.json.gz"))
        print(f"Procesando {len(archivos)} paginas del market")

        filas, corruptas = [], []
        for archivo in archivos:
            try:
                datos = json.loads(_leer_gz(archivo))
            except (EOFError, OSError, json.JSONDecodeError,
                    UnicodeDecodeError) as e:
                corruptas.append(f"{archivo} ({type(e).__name__})")
                continue

            for item in datos.get("results", []):
                desc = item.get("asset_description") or {}
                nombre = item.get("hash_name") or desc.get("market_hash_name")
                if not nombre:
                    continue

                precio_cents = item.get("sell_price")
                filas.append({
                    "market_hash_name": nombre,
                    "sell_price_usd": (precio_cents / 100
                                       if precio_cents else None),
                    "sell_listings": item.get("sell_listings"),
                })

        # Se juntan todas antes de fallar, en vez de saltearlas: saltearlas
        # pierde filas en silencio, y el piso del 95 % solo lo detecta si
        # faltan muchas.
        if corruptas:
            raise RuntimeError(
                "Paginas corruptas en el bronce:\n  " + "\n  ".join(corruptas)
                + "\nRelanzar desde descargar_precios_steam: las vuelve a pedir.")

        df = pd.DataFrame(filas)
        crudas = len(df)
        print(f"Items crudos leidos de todas las paginas: {crudas}")

        antes = len(df)
        df = df.drop_duplicates(subset="market_hash_name", keep="first")
        print(f"Duplicados por paginacion en vivo eliminados: {antes - len(df)}")

        # Un item sin precio no sirve para predecir precio.
        antes = len(df)
        df = df.dropna(subset=["sell_price_usd"])
        df = df[df["sell_price_usd"] > 0]
        if antes != len(df):
            print(f"Items sin precio o con precio <= 0 eliminados: "
                  f"{antes - len(df)}")

        ruta_salida = CARPETA_SILVER / "precios.csv"
        df.to_csv(ruta_salida, index=False, encoding="utf-8")
        print(f"Precios guardados en: {ruta_salida}")

        return {**info_descarga, "ruta": str(ruta_salida), "crudas": crudas}

    # ---------------------------------------------------------- TAREA 5
    @task
    def unir_y_validar(ruta_atributos: str, info_precios: dict, **context) -> dict:
        """**Validacion**: une las dos fuentes y chequea. Si algo falla, no se
        publica dataset.

        El chequeo de completitud es **relativo a lo que declara la fuente**, no
        un piso fijo de filas: lo que hay que atrapar es la falla silenciosa del
        scraping (una pagina vacia antes de tiempo, un 429 que corto la
        descarga), porque en ese caso el DAG termina en verde y publica un CSV
        truncado, que es peor que no publicar nada. Se exige el 95 % de lo
        planificado, medido sobre los items **crudos** y no sobre las filas
        finales: los filtros de alcance sacan filas a proposito y mezclarlo
        haria que un filtro que funciona bien parezca una descarga rota.
        """
        snapshot = info_precios["snapshot"]
        momento = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

        atributos = pd.read_csv(ruta_atributos)
        precios = pd.read_csv(info_precios["ruta"])
        print(f"Atributos: {len(atributos)} filas")
        print(f"Precios: {len(precios)} filas")

        df = atributos.merge(precios, on="market_hash_name", how="inner")
        print(f"Dataset unido: {len(df)} filas")

        # --- trazabilidad: de cuando es cada fila -------------------------
        # No es cosmetico: el criterio "mezcla de tipos" pide numericas,
        # categoricas Y fechas, y sin columna de fecha el dataset no las tiene.
        df["fecha_snapshot"] = pd.to_datetime(snapshot)
        df["captured_at"] = pd.to_datetime(momento)

        # --- antiguedad de la coleccion -----------------------------------
        # Se calcula aca y no en la capa plata porque aca existe el snapshot:
        # asi el numero es reproducible y no depende de cuando se corra.
        df["collection_release"] = pd.to_datetime(df["collection_release"],
                                                  errors="coerce")
        df["caja_fecha"] = pd.to_datetime(df["caja_fecha"], errors="coerce")
        df["antiguedad_col_dias"] = (df["fecha_snapshot"]
                                     - df["collection_release"]).dt.days

        # --- columnas de enriquecimiento 100 % vacias ----------------------
        # Vienen de una fuente que declaramos opcional: si no aportaron nada se
        # descartan con un aviso, en vez de romper el criterio "sin columnas
        # vacias" o de tirar abajo una corrida por tres features.
        vacias_opcionales = [c for c in COLUMNAS_ENRIQUECIMIENTO
                             if c in df.columns and df[c].isna().all()]
        if vacias_opcionales:
            print(f"AVISO: el catalogo no aporto {vacias_opcionales}. Se "
                  "descartan esas columnas; el dataset sigue siendo valido.")
            df = df.drop(columns=vacias_opcionales)

        problemas = []

        # --- completitud del scraping --------------------------------------
        planificados = int(info_precios.get("planificados") or 0)
        crudas = int(info_precios.get("crudas") or 0)
        if planificados:
            piso = int(planificados * 0.95)
            print(f"Completitud: {crudas} items en bronce contra "
                  f"{planificados} planificados (piso 95% = {piso})")
            if crudas < piso:
                problemas.append(
                    f"corrida incompleta: {crudas} items en bronce, se "
                    f"planificaron {planificados} (piso {piso}). Falta bronce "
                    "o la descarga se corto antes de tiempo")

        # --- cobertura del join con el catalogo de colecciones -------------
        # La coleccion paso a ser una columna del modelo: si el endpoint cambia
        # de forma queremos enterarnos en rojo, no publicar la columna vacia.
        cobertura_col = df["collection"].notna().mean()
        print(f"Cobertura de collection: {cobertura_col:.1%}")
        if cobertura_col < 0.95:
            problemas.append(
                f"solo el {cobertura_col:.1%} de las filas tiene coleccion: el "
                "join con collections.json fallo o el endpoint cambio de forma")

        # --- la clave de agrupamiento tiene que estar ----------------------
        if "skin_id" not in df.columns or df["skin_id"].isna().any():
            problemas.append("falta skin_id: sin la clave de agrupamiento no se "
                             "puede validar por grupo en la Entrega 3")

        # --- la clave: el test operativo de la unidad de analisis -----------
        if not df["market_hash_name"].is_unique:
            n = int(df["market_hash_name"].duplicated().sum())
            problemas.append(f"{n} market_hash_name repetidos: la unidad de "
                             "analisis o el pipeline estan mal")

        # --- volumen -------------------------------------------------------
        # El piso de la consigna solo se exige si el plan apuntaba a superarlo:
        # una corrida de humo con filas_objetivo=100 no tiene por que fallar.
        if planificados >= 1000 and len(df) <= 1000:
            problemas.append(f"{len(df)} filas <= 1000 requeridas por la consigna")

        # --- ancho ---------------------------------------------------------
        if df.shape[1] < 5:
            problemas.append(f"{df.shape[1]} columnas < 5 requeridas")

        # --- nulos donde no puede haberlos ---------------------------------
        obligatorias = ["market_hash_name", "weapon", "category_name",
                        "sell_price_usd", "fecha_snapshot"]
        for c in obligatorias:
            if c in df.columns and df[c].isna().any():
                problemas.append(f"{c} tiene {int(df[c].isna().sum())} nulos")

        # --- rangos --------------------------------------------------------
        if not (df["sell_price_usd"] > 0).all():
            problemas.append("hay precios <= 0")
        for c in ("min_float", "max_float"):
            if c in df.columns:
                v = df[c].dropna()
                if len(v) and not v.between(0, 1).all():
                    problemas.append(f"{c} fuera del rango 0-1")

        # --- ninguna columna 100 % vacia -----------------------------------
        vacias = [c for c in df.columns if df[c].isna().all()]
        if vacias:
            problemas.append(f"columnas 100% nulas: {vacias}")

        if problemas:
            raise ValueError("El dataset no pasa los criterios de la consigna:"
                             "\n  - " + "\n  - ".join(problemas))

        print(f"\nOK - Validacion pasada: {len(df)} filas x {df.shape[1]} "
              "columnas. Guardando CSV final.")

        ruta_std = CARPETA_INCLUDE / f"cs2_skins_dataset_{snapshot}.csv"
        df.to_csv(ruta_std, index=False, encoding="utf-8")
        print(f"CSV estandar guardado en: {ruta_std}")

        ruta_excel = CARPETA_INCLUDE / f"cs2_skins_dataset_{snapshot}_excel.csv"
        df.to_csv(ruta_excel, index=False, sep=";", decimal=",",
                  encoding="utf-8-sig")
        print(f"CSV para Excel guardado en: {ruta_excel}")

        return {
            "ruta_csv": str(ruta_std),
            "snapshot": snapshot,
            "datos_frescos": True,
            "origen": "Steam (descarga de esta corrida)",
            "declarados": info_precios.get("declarados"),
            "planificados": planificados,
            "es_muestra": info_precios.get("es_muestra"),
        }

    # ---------------------------------------------------------- TAREA 6 (FALLBACK)
    @task(trigger_rule="one_failed")
    def usar_snapshot_previo(**context) -> dict:
        """RAMA DE RESPALDO: se dispara si alguna tarea anterior fallo.

        Depende de TODAS las tareas de la rama principal (ver el grafo al
        final): sin esas dependencias `one_failed` nunca se cumpliria y el
        respaldo quedaria decorativo.

        Si no hay respaldo disponible se saltea (no falla) para que el error
        real de la rama principal quede visible.
        """
        snapshot = _snapshot(context)
        anterior = _buscar_snapshot_previo(snapshot)

        if anterior is None:
            print("No hay ningun snapshot previo disponible.")
            print("La rama de respaldo se saltea. Ver el error real en la "
                  "rama principal.")
            raise AirflowSkipException("Sin snapshot previo para el respaldo")

        ruta_destino = CARPETA_INCLUDE / f"cs2_skins_dataset_{snapshot}.csv"
        shutil.copy(anterior, ruta_destino)

        print("=" * 60)
        print("ATENCION: MODO RESPALDO ACTIVADO")
        print("=" * 60)
        print(f"El pipeline principal fallo. Se usa como respaldo: {anterior.name}")
        print(f"Copiado a: {ruta_destino}")
        print("El dataset NO son datos frescos de hoy.")

        return {
            "ruta_csv": str(ruta_destino),
            "snapshot": snapshot,
            "datos_frescos": False,
            "origen": f"RESPALDO: {anterior.name}, NO son datos de hoy",
        }

    # ---------------------------------------------------------- TAREA 7
    @task(trigger_rule="one_success")
    def generar_calidad(info_fresca: dict, info_respaldo: dict, **context) -> str:
        """Los siete criterios de la catedra, medidos sobre la salida.

        Existe para la Entrega 1: las instrucciones piden llegar con los siete
        criterios verificados, con los numeros a mano. Quedan en el log de la
        corrida y en un JSON, asi nadie abre un notebook cinco minutos antes.
        """
        info = info_fresca or info_respaldo
        if not info:
            raise ValueError("Ninguna rama produjo un dataset")

        ruta_csv = info["ruta_csv"]
        snapshot = info["snapshot"]
        datos_frescos = info["datos_frescos"]

        # parse_dates: sin esto las columnas de fecha vuelven como texto y el
        # criterio "mezcla de tipos" no las cuenta.
        df = pd.read_csv(ruta_csv, parse_dates=["fecha_snapshot", "captured_at",
                                                "collection_release"])

        if not datos_frescos:
            print("=" * 60)
            print("ATENCION: este reporte describe un dataset del RESPALDO,")
            print("NO una descarga fresca de Steam.")
            print("=" * 60)

        nulos_pct = (df.isna().mean().sort_values(ascending=False) * 100).round(2)
        nulos_con_valor = {k: f"{v}%" for k, v in nulos_pct.items() if v > 0}

        declarados = info.get("declarados")
        resumen = {
            "grupo": GRUPO,
            "snapshot": snapshot,
            "archivo": Path(ruta_csv).name,
            "run_id": context["dag_run"].run_id,
            # 0. procedencia: sin esta marca, un reporte del respaldo se ve
            #    identico a uno de una descarga fresca
            "datos_frescos": datos_frescos,
            "procedencia": info["origen"],
            # 1. clave sin duplicados
            "clave_primaria": "market_hash_name",
            "clave_unica": bool(df["market_hash_name"].is_unique),
            # 2. volumen
            "filas": int(len(df)),
            "volumen_suficiente_mas_de_1000": bool(len(df) > 1000),
            # 2b. si es una muestra hay que decirlo, y de que tamano
            "es_muestra": info.get("es_muestra"),
            "items_de_arma_en_el_mercado": declarados,
            "cobertura": (f"{len(df) / declarados * 100:.1f}%"
                          if declarados else None),
            "metodo_de_muestreo": (
                "sistematico: paginas repartidas a lo largo del catalogo "
                "ordenado por nombre, en proporcion al tamano de cada categoria"
                if info.get("es_muestra") else
                "ninguno: se bajo el mercado de armas completo"),
            # 3. ancho
            "columnas": int(df.shape[1]),
            "nombres_de_columnas": list(df.columns),
            # 4. mezcla de tipos
            "tipos": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
            # 5. nulos conocidos
            "nulos_por_columna": nulos_con_valor or "sin nulos",
            "nulos_esperados": NULOS_ESPERADOS,
            # 6. sin columnas vacias
            "columnas_vacias": [c for c in df.columns if df[c].isna().all()],
            # 7. objetivo
            # El pipeline publica el precio crudo. La variable objetivo del
            # modelo es el percentil del precio dentro del dia, y se calcula
            # en el notebook agrupando por fecha_snapshot: es una decision de
            # modelado, no un dato de la fuente.
            "variable_continua": "sell_price_usd",
            "variable_objetivo_del_modelo": (
                "percentil_precio: posicion del skin en el mercado de su dia, "
                "de 0 a 100. Se calcula en el notebook, agrupando por fecha"),
            "precio_min_usd": float(df["sell_price_usd"].min()),
            "precio_mediana_usd": float(df["sell_price_usd"].median()),
            "precio_max_usd": float(df["sell_price_usd"].max()),
            "una_fila_es": DESCRIPCION_UNIDAD,
            # 8. origen del item y agrupamiento (Entrega 2 / 3)
            "colecciones_distintas": int(df["collection"].nunique()),
            "origen_items": {k: f"{v:.1%}" for k, v in
                             df["origen"].value_counts(normalize=True).items()},
            "cobertura_collection": f"{df['collection'].notna().mean():.1%}",
            "cobertura_p_item": f"{df['p_item'].notna().mean():.1%}",
            "filas_por_skin_base_mediana": float(
                df.groupby("skin_id").size().median()),
            "clave_de_agrupamiento": "collection (GroupKFold en la Entrega 3)",
            "supuestos_declarados": [
                "float uniforme dentro del rango del skin (base de p_wear): "
                "parcialmente validado. El orden se sostiene (correlacion "
                "0,70), pero falla en los extremos; p_wear no entra al modelo",
                "caja_discontinuada viene de una lista curada externa, no de "
                "la API, y coincide exactamente con un corte de fecha",
            ],
        }

        print("\n" + "=" * 60)
        print("CHEQUEO DE CALIDAD")
        print("=" * 60)
        print(json.dumps(resumen, ensure_ascii=False, indent=2, default=str))

        ruta_json = CARPETA_INCLUDE / f"calidad_{snapshot}.json"
        ruta_json.write_text(
            json.dumps(resumen, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8")
        print(f"\nJSON de calidad guardado en: {ruta_json}")

        return str(ruta_json)

    # ---------------------------------------------------------- dependencias
    ruta_catalogo = descargar_catalogo()
    info_descarga = descargar_precios_steam()

    ruta_atributos = procesar_atributos(ruta_catalogo)
    info_precios = procesar_precios(info_descarga)

    info_fresca = unir_y_validar(ruta_atributos, info_precios)
    info_respaldo = usar_snapshot_previo()

    # El respaldo tiene que colgar de las tareas que vigila. Sin estas flechas
    # no tiene ninguna tarea aguas arriba, `one_failed` no se cumple nunca y el
    # respaldo queda decorativo: la corrida sale en verde y nadie se entera.
    [ruta_catalogo, info_descarga, ruta_atributos,
     info_precios, info_fresca] >> info_respaldo

    generar_calidad(info_fresca, info_respaldo)


cs2_pipeline()