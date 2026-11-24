#!/usr/bin/env python3
"""Baja una vez las imágenes de los skins para la aplicación (docs/img/).

El catálogo de ByMykel trae, para cada ítem, la URL de su imagen en la CDN de Steam. Steam
usa tres imágenes por diseño según el desgaste: una para Factory New y Minimal Wear, otra
para Field-Tested y Well-Worn y otra para Battle-Scarred. Este script:

1. baja el catálogo y une cada fila del CSV con su imagen por market_hash_name;
2. baja cada imagen distinta una sola vez, la achica a 320×240 y la guarda en webp;
3. escribe data/imagenes_AAAA-MM-DD.json: market_hash_name → archivo.

Las que ya están en docs/img/ no se vuelven a bajar, así que se puede cortar y retomar.

    python imagenes.py
    python imagenes.py --csv data/cs2_skins_dataset_AAAA-MM-DD.csv

Después, `python build_app.py` las incluye en la página.
"""
import argparse
import hashlib
import io
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parent
URL_CATALOGO = "https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en/skins_not_grouped.json"
TAMANO = (320, 240)
CABECERA = {"User-Agent": "Mozilla/5.0 (proyecto academico de ciencia de datos)"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--csv", default=str(RAIZ / "data" / "cs2_skins_dataset_2026-10-30.csv"))
ap.add_argument("--carpeta", default=str(RAIZ / "docs" / "img" / "skins"))
ap.add_argument("--hilos", type=int, default=32,
                help="la CDN tarda unos segundos por imagen: con muchos hilos en paralelo baja en minutos")
args = ap.parse_args()


def bajar(url, intentos=4):
    for i in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=CABECERA), timeout=40) as r:
                return r.read()
        except Exception:
            if i == intentos - 1:
                raise
            time.sleep(2 * (i + 1))


ruta = Path(args.csv)
snapshot = re.search(r"(\d{4}-\d{2}-\d{2})", ruta.name).group(1)
df = pd.read_csv(ruta, usecols=["market_hash_name"])
print("catálogo de ByMykel…", flush=True)
catalogo = json.loads(bajar(URL_CATALOGO))
imagen_de = {x["market_hash_name"]: x.get("image") for x in catalogo if x.get("market_hash_name")}
df["url"] = df["market_hash_name"].map(imagen_de)
print(f"{df['url'].notna().sum()} de {len(df)} filas con imagen; {df['url'].nunique()} imágenes distintas")

# El nombre del archivo sale de la URL: la misma imagen siempre cae en el mismo archivo.
archivo_de = {u: hashlib.sha1(u.encode()).hexdigest()[:12] + ".webp" for u in df["url"].dropna().unique()}
carpeta = Path(args.carpeta)
carpeta.mkdir(parents=True, exist_ok=True)
def sana(archivo):
    """Una imagen a medio escribir (si se cortó el script) no cuenta como bajada."""
    try:
        with Image.open(archivo) as im:
            im.verify()
        return True
    except Exception:
        return False


faltan = [u for u, a in archivo_de.items() if not ((carpeta / a).exists() and sana(carpeta / a))]
print(f"faltan bajar {len(faltan)}")


def guardar(url):
    im = Image.open(io.BytesIO(bajar(url))).convert("RGBA")
    im.thumbnail(TAMANO, Image.LANCZOS)
    temporal = carpeta / (archivo_de[url] + ".tmp")
    im.save(temporal, "WEBP", quality=80, method=6)
    temporal.replace(carpeta / archivo_de[url])   # se escribe aparte y se renombra: nunca queda a medias


errores = []
with ThreadPoolExecutor(max_workers=args.hilos) as pool:
    tareas = {pool.submit(guardar, u): u for u in faltan}
    for k, t in enumerate(as_completed(tareas), start=1):
        if t.exception():
            errores.append(tareas[t])
        if k % 250 == 0:
            print(f"  {k}/{len(faltan)}", flush=True)
if errores:
    print(f"AVISO: {len(errores)} imágenes no se pudieron bajar; volvé a correr el script para reintentarlas")

mapa = {n: archivo_de[u] for n, u in zip(df["market_hash_name"], df["url"])
        if isinstance(u, str) and (carpeta / archivo_de[u]).exists()}
salida = ruta.with_name(f"imagenes_{snapshot}.json")
salida.write_text(json.dumps(mapa, ensure_ascii=False, indent=0), encoding="utf-8")
peso = sum(f.stat().st_size for f in carpeta.glob("*.webp")) / 1e6
print(f"{len(mapa)} filas con imagen → {salida.name}; {len(list(carpeta.glob('*.webp')))} archivos, {peso:.1f} MB")
