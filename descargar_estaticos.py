#!/usr/bin/env python3
"""Baja los datos estáticos del juego que usa la aplicación (no cambian con el mercado).

Fuente: el catálogo público de ByMykel (github.com/ByMykel/CSGO-API), el mismo que usa el
pipeline para los atributos. De ahí salen el nombre, la rareza, la colección, la caja y la
imagen oficial de cada skin, y el ícono de cada colección.

    python descargar_estaticos.py               # catálogo, íconos de colecciones y armas sin diseño
    python imagenes.py                          # las fotos de cada skin, por desgaste (docs/img/skins/)

Qué escribe:
  data/estaticos/catalogo.json      un resumen chico del catálogo, sólo lo que usa la app
  docs/img/colecciones/*.png|svg    el ícono de cada colección (viene de GitHub)
  docs/img/armas/<arma>.png         cada arma sin diseño (la «default»), para «Probá el modelo»

Si una imagen no se pudo bajar, la página la pide a Steam cuando alguien la abre.
"""
import argparse
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
API = "https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en/{}.json"
DESTINO = RAIZ / "data" / "estaticos"
IMG_COL = RAIZ / "docs" / "img" / "colecciones"
IMG_ARMA = RAIZ / "docs" / "img" / "armas"


def bajar(url, intentos=3):
    req = urllib.request.Request(url, headers={"User-Agent": "cs2-proyecto-utn/1.0"})
    for i in range(intentos):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - se reintenta y al final se informa
            if i == intentos - 1:
                raise
            print(f"  reintento {url[:70]}… ({e})")


def nombre_icono(url):
    return url.rsplit("/", 1)[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(RAIZ / "data" / "cs2_skins_dataset_2026-10-30.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=["skin_id", "collection", "caja", "weapon"])
    ids, colecciones, cajas = set(df["skin_id"]), set(df["collection"].dropna()), set(df["caja"].dropna())

    print("catálogo de ByMykel…")
    skins = json.loads(bajar(API.format("skins")))
    cols = json.loads(bajar(API.format("collections")))
    crates = json.loads(bajar(API.format("crates")))
    llaves = json.loads(bajar(API.format("keys")))
    bases = json.loads(bajar(API.format("base_weapons")))
    armas = {b["name"]: b.get("image") or "" for b in bases if b["name"] in set(df["weapon"])}

    diseños = {}
    for s in skins:
        if s["id"] not in ids:
            continue
        diseños[s["id"]] = {"nombre": s["name"], "arma": s["weapon"]["name"], "diseno": s["pattern"]["name"],
                            "rareza": s["rarity"]["name"], "color": s["rarity"]["color"],
                            "float_min": s.get("min_float"), "float_max": s.get("max_float"),
                            "imagen": s.get("image") or ""}
    faltan = ids - set(diseños)
    print(f"  diseños: {len(diseños)} de {len(ids)}" + (f" (faltan {len(faltan)})" if faltan else ""))

    col_out = {}
    for c in cols:
        if c["name"] not in colecciones:
            continue
        col_out[c["name"]] = {"fecha": c.get("release_date"), "icono": c.get("image") or "",
                              "cajas": [k["name"] for k in c.get("crates", [])]}
    caja_out = {}
    for k in crates:
        if k.get("name") in cajas:
            caja_out[k["name"]] = {"imagen": k.get("image") or "", "fecha": k.get("first_sale_date")}
    # la llave que abre cada caja (para la animación de apertura)
    for l in llaves:
        for k in l.get("crates", []):
            if k.get("name") in caja_out:
                caja_out[k["name"]]["llave"] = {"nombre": l["name"], "imagen": l.get("image") or ""}

    DESTINO.mkdir(parents=True, exist_ok=True)
    (DESTINO / "catalogo.json").write_text(json.dumps(
        {"fuente": "github.com/ByMykel/CSGO-API", "disenos": diseños, "colecciones": col_out, "cajas": caja_out,
         "armas": armas},
        ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"  escrito {DESTINO / 'catalogo.json'}")

    print("íconos de colecciones…")
    IMG_COL.mkdir(parents=True, exist_ok=True)
    ok = 0
    for nombre, c in col_out.items():
        if not c["icono"]:
            continue
        archivo = IMG_COL / nombre_icono(c["icono"])
        if not archivo.exists():
            try:
                archivo.write_bytes(bajar(c["icono"]))
            except Exception as e:  # noqa: BLE001
                print(f"  sin ícono para {nombre}: {e}")
                continue
        ok += 1
    print(f"  {ok} íconos en {IMG_COL}")

    print("armas sin diseño…")
    from PIL import Image
    IMG_ARMA.mkdir(parents=True, exist_ok=True)
    hechas = 0
    for nombre, url in armas.items():
        archivo = IMG_ARMA / (re.sub(r"[^a-z0-9]+", "-", nombre.lower()).strip("-") + ".png")
        if archivo.exists() or not url:
            hechas += archivo.exists()
            continue
        try:
            im = Image.open(io.BytesIO(bajar(url + ("/360fx360f" if "steamstatic" in url else "")))).convert("RGBA")
            im.thumbnail((360, 270))
            im.save(archivo, optimize=True)
            hechas += 1
        except Exception as e:  # noqa: BLE001
            print(f"  {nombre}: no se pudo bajar ({e}); la página la pide a Steam")
    print(f"  {hechas} de {len(armas)} armas en {IMG_ARMA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
