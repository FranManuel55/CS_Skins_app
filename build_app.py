#!/usr/bin/env python3
"""Construye la aplicación web del proyecto (docs/index.html).

Toma el CSV que publica el pipeline, reproduce el modelo de la Entrega 3 (versión V2,
elegido con R²) con la misma semilla, calcula todo lo que muestra la aplicación y
escribe una sola página HTML con los gráficos de Altair y los datos adentro.

    python build_app.py                                  # snapshot del 30/10
    python build_app.py --csv data/cs2_skins_dataset_AAAA-MM-DD.csv

Si los datos cambian, se vuelve a correr y la aplicación queda actualizada.
Requiere scikit-learn 1.6.1 (la versión del notebook).
"""
import argparse
import json
import re
from datetime import date
from functools import partial
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RAIZ = Path(__file__).resolve().parent
SEMILLA = 42

# ----------------------------------------------------------------- argumentos
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--csv", default=str(RAIZ / "data" / "cs2_skins_dataset_2026-10-30.csv"))
ap.add_argument("--plantilla", default=str(RAIZ / "app" / "plantilla.html"))
ap.add_argument("--salida", default=str(RAIZ / "docs" / "index.html"))
ap.add_argument("--verificar", default=str(RAIZ / "data" / "modelo_metadatos_notebook.json"),
                help="metadatos que guardó el notebook, para comprobar que los números coinciden")
args = ap.parse_args()

if sklearn.__version__ != "1.6.1":
    print(f"AVISO: scikit-learn {sklearn.__version__}; el notebook usa 1.6.1 y los números pueden diferir.")
np.random.seed(SEMILLA)

# ================================================================ 1 · datos
ruta = Path(args.csv)
SNAPSHOT = re.search(r"(\d{4}-\d{2}-\d{2})", ruta.name).group(1)
df = pd.read_csv(ruta)
for c in ("fecha_snapshot", "captured_at", "collection_release", "caja_fecha"):
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")
for c in ("stattrak", "souvenir"):
    df[c] = df[c].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
if "antiguedad_col_dias" not in df.columns or df["antiguedad_col_dias"].isna().all():
    df["antiguedad_col_dias"] = (df["fecha_snapshot"] - df["collection_release"]).dt.days
print(f"snapshot {SNAPSHOT}: {len(df)} filas")

calidad_ruta = ruta.with_name(f"calidad_{SNAPSHOT}.json")
CALIDAD = json.loads(calidad_ruta.read_text(encoding="utf-8")) if calidad_ruta.exists() else {}
embudo_ruta = ruta.with_name(f"embudo_{SNAPSHOT}.json")
EMBUDO = json.loads(embudo_ruta.read_text(encoding="utf-8")) if embudo_ruta.exists() else {}

COL_ORDINALES = ["rarity_orden", "wear_orden"]
COL_BOOLEANAS = ["stattrak", "souvenir"]
COL_CATEGORICAS = ["weapon", "origen"]
COL_PROB_DROP = ["neg_log_p_item"]
COLUMNAS_MODELO = COL_ORDINALES + COL_BOOLEANAS + COL_CATEGORICAS + COL_PROB_DROP
# Decisión de la sección 7 del notebook: con R², sacar cualquiera de las dos columnas en
# duda (origen, neg_log_p_item) baja el R² por más que el desvío, así que quedan las siete.
COLUMNAS_FINALES = list(COLUMNAS_MODELO)

# ================================================================ 2 · partición (sección 3 del notebook)
FRACCION_TEST = 0.20
orden_col = (df.dropna(subset=["collection_release"]).groupby("collection")["collection_release"]
               .first().sort_values(ascending=False))
filas_acum = df["collection"].value_counts().reindex(orden_col.index).cumsum()
n_col = int((filas_acum < FRACCION_TEST * len(df)).sum()) + 1
COLECCIONES_TEST = set(orden_col.index[:n_col])
es_test = df["collection"].isin(COLECCIONES_TEST)
train = df[~es_test].reset_index(drop=True)
test = df[es_test].reset_index(drop=True)


def percentil_contra(precios, mercado):
    mercado = np.sort(np.asarray(mercado, dtype=float))
    return np.searchsorted(mercado, np.asarray(precios, dtype=float), side="right") / len(mercado) * 100


MERCADO = np.sort(train["sell_price_usd"].to_numpy(dtype=float))
train["percentil"] = percentil_contra(train["sell_price_usd"], MERCADO)
test["percentil"] = percentil_contra(test["sell_price_usd"], MERCADO)
y_tr, y_te = train["percentil"], test["percentil"]
COLUMNAS_X = COLUMNAS_MODELO + ["antiguedad_col_dias", "rarity", "wear"]
X_tr, X_te = train[COLUMNAS_X], test[COLUMNAS_X]
assert not {"percentil", "sell_price_usd", "sell_listings"} & set(X_tr.columns)


def pedazos_temporales(datos, n_bloques=4, fraccion_inicial=0.40):
    fecha = datos.groupby("collection")["collection_release"].first().dropna().sort_values()
    filas = datos["collection"].value_counts().reindex(fecha.index)
    acumulado = (filas.cumsum() / filas.sum()).to_numpy()
    cortes = [fraccion_inicial + (1 - fraccion_inicial) * k / n_bloques for k in range(n_bloques + 1)]
    bloque_de = dict(zip(fecha.index, np.searchsorted(cortes, acumulado, side="left")))
    bloque = datos["collection"].map(bloque_de).fillna(0).astype(int).to_numpy()
    return [(np.where(bloque < k)[0], np.where(bloque == k)[0]) for k in range(1, n_bloques + 1)]


PEDAZOS = pedazos_temporales(train)
tabla_pedazos = []
for i, (i_fit, i_val) in enumerate(PEDAZOS, start=1):
    f_val = train.iloc[i_val]
    tabla_pedazos.append({"pedazo": i, "entrena": len(i_fit), "valida": len(i_val),
                          "desde": f"{f_val['collection_release'].min():%Y}",
                          "hasta": f"{f_val['collection_release'].max():%Y}"})

# ================================================================ 3 · pipeline, métricas, pisos
def preprocesamiento(columnas):
    pasos = []
    numericas = [c for c in COL_ORDINALES + ["antiguedad_col_dias"] if c in columnas]
    if numericas:
        pasos.append(("ordinales", Pipeline([("imputar", SimpleImputer(strategy="median")),
                                             ("escalar", StandardScaler())]), numericas))
    if "neg_log_p_item" in columnas:
        pasos.append(("prob_drop", Pipeline([("imputar", SimpleImputer(strategy="median", add_indicator=True)),
                                             ("escalar", StandardScaler())]), ["neg_log_p_item"]))
    booleanas = [c for c in COL_BOOLEANAS if c in columnas]
    if booleanas:
        pasos.append(("booleanas", "passthrough", booleanas))
    categoricas = [c for c in COL_CATEGORICAS if c in columnas]
    if categoricas:
        pasos.append(("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas))
    return ColumnTransformer(pasos, remainder="drop")


RECORTE = partial(np.clip, a_min=0, a_max=100)


def armar_modelo(regresor, columnas=None):
    columnas = COLUMNAS_MODELO if columnas is None else columnas
    pipe = Pipeline([("prep", preprocesamiento(columnas)), ("modelo", regresor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.asarray, inverse_func=RECORTE, check_inverse=False)


def sesgo(y_real, y_pred):
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_real)))


METRICAS = {"r2": "r2", "mae": "neg_mean_absolute_error", "sesgo": make_scorer(sesgo)}


def evaluar(nombre, modelo):
    r = cross_validate(modelo, X_tr, y_tr, cv=PEDAZOS, scoring=METRICAS)
    return {"modelo": nombre, "r2": float(r["test_r2"].mean()), "desvio": float(r["test_r2"].std()),
            "mae": float(-r["test_mae"].mean()), "sesgo": float(r["test_sesgo"].mean()),
            "_pedazos": r["test_r2"]}


def comparar(base, otro):
    d = base["_pedazos"] - otro["_pedazos"]
    return {"dif": float(d.mean()), "desvio_dif": float(d.std()), "gana": int((d > 0).sum())}


class MediaPorGrupo(RegressorMixin, BaseEstimator):
    def __init__(self, columnas=("rarity",)):
        self.columnas = columnas

    def fit(self, X, y):
        cols = list(self.columnas)
        d = X[cols].copy()
        d["_y"] = np.asarray(y, dtype=float)
        self.tabla_ = d.groupby(cols)["_y"].mean()
        self.media_global_ = float(np.mean(d["_y"]))
        return self

    def predict(self, X):
        cols = list(self.columnas)
        claves = pd.MultiIndex.from_frame(X[cols]) if len(cols) > 1 else X[cols[0]]
        return pd.Series(self.tabla_.reindex(claves).to_numpy()).fillna(self.media_global_).to_numpy()


print("pisos y comparación de modelos…")
pisos = [evaluar("Piso: el promedio", armar_modelo(DummyRegressor(strategy="mean"))),
         evaluar("Tabla por rareza", MediaPorGrupo(("rarity",))),
         evaluar("Tabla por rareza y exterior", MediaPorGrupo(("rarity", "wear")))]
PISO, MEJOR_PISO = pisos[0], max(pisos, key=lambda r: r["r2"])

CANDIDATOS = {
    "Regresión lineal (Ridge)": Ridge(alpha=1.0),
    "Árbol de decisión": DecisionTreeRegressor(random_state=SEMILLA),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEMILLA, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=SEMILLA),
}
REFERENCIA = "Regresión lineal (Ridge)"
comparacion = [evaluar(n, armar_modelo(m)) for n, m in CANDIDATOS.items()]
res = {r["modelo"]: r for r in comparacion}
duelos = []
for nombre in list(CANDIDATOS)[1:]:
    c = comparar(res[nombre], res[REFERENCIA])
    duelos.append({"retador": nombre, **c, "supera": c["dif"] > c["desvio_dif"]})
MEJOR_R2 = max(comparacion, key=lambda r: r["r2"])["modelo"]
ELEGIDO = MEJOR_R2
if MEJOR_R2 != REFERENCIA and not next(d for d in duelos if d["retador"] == MEJOR_R2)["supera"]:
    ELEGIDO = REFERENCIA
print("elegido:", ELEGIDO)

# ================================================================ 4 · ajuste (sección 8)
P = "regressor__modelo__"
GRILLAS = {
    "Regresión lineal (Ridge)": {P + "alpha": [0.01, 0.1, 1, 3, 10, 30, 100]},
    "Árbol de decisión": {P + "max_depth": [3, 5, 8, 12, None], P + "min_samples_leaf": [1, 5, 10, 20, 40]},
    "Random Forest": {P + "max_depth": [None, 8, 16], P + "min_samples_leaf": [1, 3, 5, 10],
                      P + "max_features": [1.0, 0.5, "sqrt"]},
    "Gradient Boosting": {P + "learning_rate": [0.03, 0.1], P + "max_leaf_nodes": [15, 31],
                          P + "min_samples_leaf": [10, 20, 40], P + "loss": ["squared_error", "absolute_error"]},
}
print("grilla…")
busqueda = GridSearchCV(armar_modelo(CANDIDATOS[ELEGIDO], COLUMNAS_FINALES), GRILLAS[ELEGIDO],
                        cv=PEDAZOS, scoring=METRICAS, refit="r2", n_jobs=1).fit(X_tr, y_tr)
cvr = pd.DataFrame(busqueda.cv_results_)
i_mejor = busqueda.best_index_
R2_CV, DESVIO_CV = float(busqueda.best_score_), float(cvr.loc[i_mejor, "std_test_r2"])
MAE_CV, SESGO_CV = float(-cvr.loc[i_mejor, "mean_test_mae"]), float(cvr.loc[i_mejor, "mean_test_sesgo"])
ajustado = {"_pedazos": np.array([cvr.loc[i_mejor, f"split{i}_test_r2"] for i in range(len(PEDAZOS))])}
AJUSTE = comparar(ajustado, res[ELEGIDO])
MEJORES = {k.replace(P, ""): v for k, v in busqueda.best_params_.items()}
N_GRILLA = len(cvr)

# ================================================================ 5 · la prueba (sección 9)
modelo_final = busqueda.best_estimator_
test["predicho"] = modelo_final.predict(X_te)
test["error"] = test["predicho"] - test["percentil"]
piso_te = armar_modelo(DummyRegressor(strategy="mean")).fit(X_tr, y_tr).predict(X_te)
cols_piso = ("rarity", "wear") if "exterior" in MEJOR_PISO["modelo"] else ("rarity",)
piso_inf_te = MediaPorGrupo(cols_piso).fit(X_tr, y_tr).predict(X_te)
R2_TEST, MAE_TEST, SESGO_TEST = r2_score(y_te, test["predicho"]), mean_absolute_error(y_te, test["predicho"]), sesgo(y_te, test["predicho"])
PRUEBA = [
    {"modelo": "Piso: el promedio", "validacion": PISO["r2"], "prueba": r2_score(y_te, piso_te),
     "sesgo_prueba": sesgo(y_te, piso_te)},
    {"modelo": "Tabla por rareza y exterior", "validacion": MEJOR_PISO["r2"], "prueba": r2_score(y_te, piso_inf_te),
     "sesgo_prueba": sesgo(y_te, piso_inf_te)},
    {"modelo": ELEGIDO, "validacion": R2_CV, "prueba": R2_TEST, "sesgo_prueba": SESGO_TEST},
]
por_col = test.groupby("collection")["error"].mean()
COLECCIONES_SESGO_POS = int((por_col > 0).sum())

corrimiento = []
for o in ["caja", "drop de mapa", "paquete souvenir", "sin identificar"]:
    a, b = train[train["origen"] == o], test[test["origen"] == o]
    corrimiento.append({"origen": o, "lado": "entrenamiento", "mediana": float(a["percentil"].median()), "filas": len(a)})
    corrimiento.append({"origen": o, "lado": "prueba", "mediana": float(b["percentil"].median()), "filas": len(b),
                        "sesgo": float(b["error"].mean())})
mapa_tr = train[train["origen"] == "drop de mapa"]
mapa_te = test[test["origen"] == "drop de mapa"]
MAPA = {"col_tr": int(mapa_tr["collection"].nunique()), "col_te": int(mapa_te["collection"].nunique()),
        "ultima_tr": f"{mapa_tr['collection_release'].max():%m/%Y}",
        "primera_te": f"{mapa_te['collection_release'].min():%m/%Y}"}

TRAMOS = [f"{a}–{a + 10}" for a in range(0, 100, 10)]
test["tramo"] = pd.cut(test["percentil"], bins=range(0, 101, 10), labels=TRAMOS, include_lowest=True)


def percentil_a_dolares(p, mercado=MERCADO):
    return np.quantile(mercado, np.clip(p, 0, 100) / 100)


test["precio_predicho"] = percentil_a_dolares(test["predicho"].to_numpy())
test["error_usd"] = test["precio_predicho"] - test["sell_price_usd"]
test["veces"] = 10 ** (np.log10(test["precio_predicho"]) - np.log10(test["sell_price_usd"])).abs()
USD = {
    "real_mediano": float(test["sell_price_usd"].median()),
    "pred_mediano": float(test["precio_predicho"].median()),
    "abs_medio": float(test["error_usd"].abs().mean()),
    "abs_mediano": float(test["error_usd"].abs().median()),
    "veces_mediana": float(test["veces"].median()),
    "dentro_x2": float(100 * (test["veces"] <= 2).mean()),
    "mas_caro": float(100 * (test["error_usd"] > 0).mean()),
}
por_tramo = (test.groupby("tramo", observed=True)
                 .agg(filas=("error", "size"), mae=("error", lambda e: e.abs().mean()), sesgo=("error", "mean"),
                      real=("sell_price_usd", "median"), pred=("precio_predicho", "median"),
                      usd=("error_usd", lambda e: e.abs().median()))
                 .reset_index())
peores = (test.reindex(test["error"].abs().sort_values(ascending=False).index)
              .head(6)[["market_hash_name", "collection", "origen", "percentil", "predicho",
                        "sell_price_usd", "precio_predicho"]])

print("importancia por permutación…")
imp = permutation_importance(modelo_final, X_te[COLUMNAS_FINALES], y_te, scoring="r2", n_repeats=20,
                             random_state=SEMILLA)
NOMBRES_COL = {"rarity_orden": "Rareza", "wear_orden": "Exterior", "stattrak": "StatTrak", "souvenir": "Souvenir",
               "weapon": "Arma", "origen": "Origen", "neg_log_p_item": "Dificultad de drop"}
IMPORTANCIA = sorted([{"columna": NOMBRES_COL[c], "baja": float(m), "desvio": float(s)}
                      for c, m, s in zip(COLUMNAS_FINALES, imp.importances_mean, imp.importances_std)],
                     key=lambda r: -r["baja"])

# ================================================================ 6 · predicciones para explorar
# Prueba: el modelo evaluado de la sección 9. Entrenamiento: un modelo igual que no vio la
# colección del skin (validación cruzada por colección, 5 grupos, con los mismos hiperparámetros).
print("predicciones fuera de muestra para explorar…")
oof = np.zeros(len(train))
for i_fit, i_val in GroupKFold(n_splits=5).split(X_tr, y_tr, groups=train["collection"].fillna("sin colección")):
    m = clone(modelo_final).fit(X_tr.iloc[i_fit], y_tr.iloc[i_fit])
    oof[i_val] = m.predict(X_tr.iloc[i_val])
train["predicho"] = oof
R2_OOF = r2_score(y_tr, oof)

# ================================================================ 7 · el modelo de la aplicación (sección 12)
MERCADO_TOTAL = np.sort(df["sell_price_usd"].to_numpy(dtype=float))
df["q"] = percentil_contra(df["sell_price_usd"], MERCADO_TOTAL)
modelo_app = clone(modelo_final).fit(df[COLUMNAS_X], df["q"])

RAREZAS = ["Consumer Grade", "Industrial Grade", "Mil-Spec Grade", "Restricted", "Classified", "Covert"]
EXTERIORES = ["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"]
ARMAS = sorted(df["weapon"].unique())
ORIGENES = ["caja", "drop de mapa", "paquete souvenir", "sin identificar"]
VARIANTES = {"caja": ["Normal", "StatTrak"], "drop de mapa": ["Normal", "StatTrak"],
             "paquete souvenir": ["Souvenir"], "sin identificar": ["Souvenir"]}
NIVELES = [("fácil", 0.15), ("típica", 0.5), ("difícil", 0.85)]   # cuantiles de neg_log_p_item

caja_tr = df[df["origen"] == "caja"]
filas = []
for o in ORIGENES:
    for var in VARIANTES[o]:
        st, sv = int(var == "StatTrak"), int(var == "Souvenir")
        for r_i, r in enumerate(RAREZAS):
            for w_i, w in enumerate(EXTERIORES):
                if o == "caja":
                    g = caja_tr[(caja_tr["rarity"] == r) & (caja_tr["wear"] == w) & (caja_tr["stattrak"] == st)]
                    base_g = g if len(g) >= 5 else caja_tr
                    niveles = [(n, float(base_g["neg_log_p_item"].quantile(qq))) for n, qq in NIVELES]
                else:
                    niveles = [("—", np.nan)]
                for n_i, (nivel, nl) in enumerate(niveles):
                    for a in ARMAS:
                        filas.append({"rarity_orden": r_i, "wear_orden": w_i, "stattrak": st, "souvenir": sv,
                                      "weapon": a, "origen": o, "neg_log_p_item": nl,
                                      "antiguedad_col_dias": np.nan, "rarity": r, "wear": w,
                                      "nivel": n_i if o == "caja" else 1})
grilla = pd.DataFrame(filas)
grilla["pred"] = modelo_app.predict(grilla[COLUMNAS_X])
# Comprobación: la grilla da lo mismo que el modelo sobre un caso armado a mano.
caso = grilla.iloc[[123]]
assert np.isclose(modelo_app.predict(caso[COLUMNAS_X])[0], caso["pred"].iloc[0])

visto_ar = df.groupby(["weapon", "rarity_orden"]).size()
visto_or = df.groupby(["origen", "rarity_orden"]).size()
visto_ov = df.groupby(["origen", "stattrak"]).size()
# Cuántos ítems de caja respaldan cada arma+rareza+exterior+variante: la rebanada exacta
# que mueve "Dificultad de sacarlo de la caja". Mediana real: 4 filas; con tan pocas, el
# nivel elegido no tiene de dónde sacar una relación confiable para esa arma puntual.
visto_an = caja_tr.groupby(["weapon", "rarity_orden", "wear_orden", "stattrak"]).size()
GRILLA = {}
for row in grilla.itertuples(index=False):
    clave = f"{ARMAS.index(row.weapon)}|{row.rarity_orden}|{row.wear_orden}|{ORIGENES.index(row.origen)}|{row.stattrak}|{row.nivel}"
    GRILLA[clave] = round(float(row.pred), 1)
VISTOS = {
    "arma_rareza": {f"{ARMAS.index(a)}|{r}": int(n) for (a, r), n in visto_ar.items()},
    "origen_rareza": {f"{ORIGENES.index(o)}|{r}": int(n) for (o, r), n in visto_or.items()},
    "origen_stattrak": {f"{ORIGENES.index(o)}|{s}": int(n) for (o, s), n in visto_ov.items()},
    "arma_nivel": {f"{ARMAS.index(a)}|{r}|{w}|{st}": int(n) for (a, r, w, st), n in visto_an.items()},
}
CUANTILES_PRECIO = [round(float(np.quantile(MERCADO_TOTAL, k / 1000)), 2) for k in range(1001)]

# ================================================================ 8 · hipótesis de la Entrega 2, recalculadas
base = df[(df["stattrak"] == 0) & (df["souvenir"] == 0)]
sv = df[df["souvenir"] == 1].groupby(["skin_id", "wear_orden"])["q"].first()
nr = base.groupby(["skin_id", "wear_orden"])["q"].first()
pares_sv = pd.concat([sv.rename("sv"), nr.rename("nr")], axis=1, join="inner")
dif_sv = pares_sv["sv"] - pares_sv["nr"]
H2 = {"pares": len(pares_sv), "d": float(dif_sv.mean() / dif_sv.std(ddof=1)), "mediana": float(dif_sv.median())}
rhos = []
for _, g in base.groupby("skin_id"):
    if g["wear_orden"].nunique() >= 3:
        rhos.append(g["wear_orden"].corr(g["q"], method="spearman"))
rhos = pd.Series(rhos).dropna()
H3 = {"skins": len(rhos), "rho": float(rhos.median()), "baja": float(100 * (rhos < 0).mean())}
caja = df[df["origen"] == "caja"]
r4, n4 = [], []
for _, g in caja.groupby("rarity"):
    if len(g) > 30:
        r4.append(g["neg_log_p_item"].corr(g["q"], method="spearman")); n4.append(len(g))
H4 = {"r": float(np.average(r4, weights=n4)), "filas": int(sum(n4))}
MEDIANA_RAREZA = df.groupby("rarity_orden")["q"].median().to_dict()
SPEARMAN_RAREZA = float(df["rarity_orden"].corr(df["q"], method="spearman"))

# ================================================================ 9 · verificación contra el notebook
VERIFICADO = "no se encontró el archivo de metadatos del notebook"
ruta_meta = Path(args.verificar)
if ruta_meta.exists():
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    if meta.get("snapshot") == SNAPSHOT:
        for k, v in {"r2_cv": R2_CV, "r2_prueba": R2_TEST, "mae_prueba": MAE_TEST, "sesgo_prueba": SESGO_TEST}.items():
            assert abs(meta[k] - v) < 6e-3, f"{k}: notebook {meta[k]} contra aplicación {v}"
        VERIFICADO = (f"coincide con el notebook: R² {meta['r2_cv']:.3f} en validación y "
                      f"{meta['r2_prueba']:.3f} en la prueba")
    else:
        VERIFICADO = f"datos de otra fecha que el notebook ({meta.get('snapshot')}): números nuevos"
print("verificación:", VERIFICADO)

def n(x, d=0):
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x):
    return f"{n(x)} %"


# ================================================================ 10 · los grupos que cuentan la historia
# Para el modelo, todo lo que no viene de caja ni de paquete y tiene colección es «drop de mapa».
# Para contar la historia lo partimos en dos épocas: las colecciones de drop lanzadas desde la
# actualización «The Armory» (2/10/2024) se reparten de otra forma (canjes con un pase o el
# paquete semanal) y valen mucho menos. El modelo NO usa esta división: sólo la página.
FECHA_ERA = pd.Timestamp("2024-10-01")
CANAL = {
    "The Graphic Design Collection": "Armory: se canjea con créditos de un pase",
    "The Overpass 2024 Collection": "Armory: se canjea con créditos de un pase",
    "The Sport & Field Collection": "Armory: se canjea con créditos de un pase",
    "The Train 2025 Collection": "Armory: se canjea con créditos de un pase",
    "The Spy Tech Collection": "Armory: se canjea con créditos de un pase",
    "The Arabesque Collection": "Armory: se canjea con créditos de un pase",
    "The Ascent Collection": "Paquete semanal: premio para cualquier jugador Prime",
    "The Boreal Collection": "Paquete semanal: premio para cualquier jugador Prime",
    "The Radiant Collection": "Paquete semanal: premio para cualquier jugador Prime",
    "The Achroma Collection": "Paquete semanal: premio para cualquier jugador Prime",
    "The Harlequin Collection": "Paquete semanal: premio para cualquier jugador Prime",
    "The Genesis Collection": "Terminal del paquete semanal (también se compra en el mercado)",
    "The Dead Hand Collection": "Terminal del paquete semanal (también se compra en el mercado)",
    "Limited Edition Item": "Ediciones limitadas del Armory (25 créditos cada una)",
}
GRUPOS = ["caja", "drop de mapa clásico", "drop nuevo", "paquete souvenir", "sin identificar"]
NOMBRE_GRUPO = ["Caja", "Drop de mapa · antes del Armory", "Drop de mapa · era Armory", "Paquete de torneo", "Souvenir sin paquete"]
COL_GRUPO = ["#2fae9f", "#e8a33b", "#ff5f3d", "#7cc35a", "#c79b6b"]
ORIGEN_TXT = {"caja": "Caja", "drop de mapa": "Drop de mapa", "paquete souvenir": "Paquete de torneo", "sin identificar": "Souvenir sin paquete"}
COL_ORIGEN = {"caja": "#2fae9f", "drop de mapa": "#e8a33b", "paquete souvenir": "#7cc35a", "sin identificar": "#c79b6b"}


def grupo_de(origen, fecha):
    if origen == "drop de mapa":
        return "drop nuevo" if pd.notna(fecha) and fecha >= FECHA_ERA else "drop de mapa clásico"
    return origen


for d_ in (df, train, test):
    d_["grupo"] = [grupo_de(o, f) for o, f in zip(d_["origen"], d_["collection_release"])]

# ---------------------------------------------------------------- catálogo estático (descargar_estaticos.py)
ruta_cat = RAIZ / "data" / "estaticos" / "catalogo.json"
CAT = json.loads(ruta_cat.read_text(encoding="utf-8")) if ruta_cat.exists() else {"disenos": {}, "colecciones": {}, "cajas": {}}
if not CAT["disenos"]:
    print("AVISO: falta data/estaticos/catalogo.json; corré descargar_estaticos.py para tener las imágenes.")
STEAM = "https://community.akamai.steamstatic.com/economy/image/"


ruta_img = RAIZ / "data" / f"imagenes_{SNAPSHOT}.json"
IMAGEN_DE = json.loads(ruta_img.read_text(encoding="utf-8")) if ruta_img.exists() else {}
if not IMAGEN_DE:
    print("AVISO: no hay imágenes por desgaste; corré imagenes.py para tenerlas.")


def img_local(nombre_mercado):
    """Nombre del archivo (sin .webp) de la foto de ese skin en docs/img/skins/, o '' si no está."""
    f = IMAGEN_DE.get(nombre_mercado, "")
    return f[:-5] if f.endswith(".webp") and (RAIZ / "docs" / "img" / "skins" / f).exists() else ""


def img_url(url):
    """Imagen de Steam: se guarda sin el prefijo común ('s:…'); la página le agrega el tamaño."""
    return "s:" + url[len(STEAM):] if url.startswith(STEAM) else url


def img_skin(sid):
    if (RAIZ / "docs" / "img" / "skins" / f"{sid}.png").exists():
        return f"img/skins/{sid}.png"
    return img_url(CAT["disenos"].get(sid, {}).get("imagen", ""))


def img_coleccion(nombre):
    url = CAT["colecciones"].get(nombre, {}).get("icono", "")
    if url:
        local = RAIZ / "docs" / "img" / "colecciones" / url.rsplit("/", 1)[-1]
        if local.exists():
            return "img/colecciones/" + local.name
    return url


def img_caja(nombre):
    return img_url(CAT["cajas"].get(nombre, {}).get("imagen", "")) if isinstance(nombre, str) else ""


def corto(c):
    return re.sub(r"^The |\s*Collection$", "", str(c))


# ================================================================ 11 · datos que viajan con la página
lado = np.r_[np.zeros(len(train), int), np.ones(len(test), int)]
todo = pd.concat([train, test], ignore_index=True)
todo["q"] = percentil_contra(todo["sell_price_usd"], MERCADO_TOTAL)
todo["ph"] = percentil_a_dolares(todo["predicho"].to_numpy())
todo["var"] = np.select([todo["stattrak"] == 1, todo["souvenir"] == 1], [1, 2], 0)

fecha_col = df.groupby("collection")["collection_release"].first()
COLS = sorted(df["collection"].unique(), key=lambda c: (pd.isna(fecha_col[c]), fecha_col[c] if pd.notna(fecha_col[c]) else 0, c))
i_col = {c: i for i, c in enumerate(COLS)}
dis = (todo.groupby("skin_id").agg(arma=("weapon", "first"), patron=("pattern", "first"), r=("rarity_orden", "first"),
                                   c=("collection", "first")).reset_index())
dis = dis.sort_values(["c", "r", "arma"], key=lambda s: s.map(i_col) if s.name == "c" else s).reset_index(drop=True)
i_dis = {s: i for i, s in enumerate(dis["skin_id"])}
def rango_float(sid):
    c = CAT["disenos"].get(sid, {})
    a, b = c.get("float_min"), c.get("float_max")
    return [round(float(a), 2) if a is not None else 0.0, round(float(b), 2) if b is not None else 1.0]


todo["img"] = [img_local(m) for m in todo["market_hash_name"]]
IMGS = sorted({i for i in todo["img"] if i})
i_img = {im: k for k, im in enumerate(IMGS)}
# foto representativa de cada diseño: la versión normal más cercana a Field-Tested
rep_ = (todo.assign(_v=(todo["var"] != 0).astype(int), _w=(todo["wear_orden"] - 2).abs(), _s=(todo["img"] == "").astype(int))
        .sort_values(["_s", "_v", "_w"]).drop_duplicates("skin_id").set_index("skin_id")["img"])
DISENOS = [[f"{d.arma} | {d.patron}", d.arma, int(d.r), i_col[d.c],
            f"img/skins/{rep_[d.skin_id]}.webp" if rep_.get(d.skin_id) else img_skin(d.skin_id)] + rango_float(d.skin_id)
           for d in dis.itertuples()]


def img_arma(nombre):
    local = RAIZ / "docs" / "img" / "armas" / (re.sub(r"[^a-z0-9]+", "-", nombre.lower()).strip("-") + ".png")
    return "img/armas/" + local.name if local.exists() else img_url(CAT.get("armas", {}).get(nombre, ""))


ARMA_IMG = [img_arma(a) for a in ARMAS]

SKINS = pd.DataFrame({
    "d": todo["skin_id"].map(i_dis), "r": todo["rarity_orden"].astype(int), "w": todo["wear_orden"].astype(int),
    "v": todo["var"].astype(int), "g": todo["grupo"].map({g: i for i, g in enumerate(GRUPOS)}),
    "p": todo["sell_price_usd"].round(2), "q": todo["q"].round(1),
    "y": todo["percentil"].round(1), "yh": todo["predicho"].round(1), "ph": todo["ph"].round(2), "l": lado,
    "c": todo["collection"].map(i_col), "a": todo["weapon"].map({a: i for i, a in enumerate(ARMAS)}),
    "i": todo["img"].map(lambda x: i_img.get(x, -1)),
}).to_dict(orient="records")

info_col = []
for c in COLS:
    g = todo[todo["collection"] == c]
    normales = g[(g["stattrak"] == 0) & (g["souvenir"] == 0)]
    tipo = (normales if len(normales) else g)["grupo"].mode().iloc[0]
    origen_c = (normales if len(normales) else g)["origen"].mode().iloc[0]
    if c == "Limited Edition Item":      # ítems sueltos del Armory, sin fecha en el catálogo
        tipo = "drop nuevo"
    f = fecha_col[c]
    caja = g["caja"].dropna().iloc[0] if g["caja"].notna().any() else ""
    info_col.append({
        "c": i_col[c], "nombre": corto(c), "f": f"{f:%Y-%m-%d}" if pd.notna(f) else "",
        "tipo": NOMBRE_GRUPO[GRUPOS.index(tipo)], "ti": GRUPOS.index(tipo),
        "origen": ORIGEN_TXT[origen_c], "oi": ORIGENES.index(origen_c),
        "canal": CANAL.get(c, ""), "filas": len(g), "disenos": int(g["skin_id"].nunique()),
        "mediana": round(float(g["sell_price_usd"].median()), 2),
        "armas": "|" + "|".join(sorted(g["weapon"].unique())) + "|",
        "icono": img_coleccion(c), "caja": caja, "caja_img": img_caja(caja),
        "lado": "prueba" if c in COLECCIONES_TEST else "entrenamiento",
    })

# ---------------------------------------------------------------- la introducción: ejemplos reales
def fila_ejemplo(r):
    return {"d": i_dis[r.skin_id], "w": int(r.wear_orden), "v": int(r["var"]), "p": round(float(r.sell_price_usd), 2),
            "q": round(float(r.q), 1), "g": GRUPOS.index(r.grupo), "i": i_img.get(r["img"], -1)}


ak = todo[todo["weapon"] == "AK-47"].sort_values("sell_price_usd")
GANCHO = [fila_ejemplo(ak.iloc[int(round(k * (len(ak) - 1)))]) for k in (0, 0.25, 0.5, 0.75, 0.93, 1)]

PROB_CAJA = {2: 79.92, 3: 15.98, 4: 3.20, 5: 0.64}   # probabilidades publicadas por Valve (China, 2017)
POPULARES = ["AK-47", "AWP", "M4A4", "M4A1-S", "USP-S", "Glock-18", "Desert Eagle"]
RAREZA_INTRO = []
for r_i, r in enumerate(RAREZAS):
    g = todo[todo["rarity_orden"] == r_i]
    pop = g[g["weapon"].isin(POPULARES) & (g["var"] == 0)]
    base_ej = pop if len(pop) else g[g["var"] == 0]
    cuenta = base_ej.groupby("skin_id").size().sort_values(ascending=False)
    ej = base_ej[base_ej["skin_id"] == cuenta.index[0]].sort_values("wear_orden").iloc[0]
    RAREZA_INTRO.append({"mediana": round(float(g["sell_price_usd"].median()), 2), "filas": len(g),
                         "q": round(float(g["q"].median()), 0), "prob": PROB_CAJA.get(r_i), "ej": fila_ejemplo(ej)})

ext_id = dis.loc[(dis["arma"] == "AK-47") & (dis["patron"] == "Asiimov"), "skin_id"]
ext_id = ext_id.iloc[0] if len(ext_id) else todo[todo["var"] == 1]["skin_id"].iloc[0]
ext = todo[todo["skin_id"] == ext_id]
EXTERIOR_INTRO = {"d": i_dis[ext_id],
                  "normal": [float(ext[(ext["wear_orden"] == w) & (ext["var"] == 0)]["sell_price_usd"].min()) if
                             len(ext[(ext["wear_orden"] == w) & (ext["var"] == 0)]) else None for w in range(5)],
                  "stattrak": [float(ext[(ext["wear_orden"] == w) & (ext["var"] == 1)]["sell_price_usd"].min()) if
                               len(ext[(ext["wear_orden"] == w) & (ext["var"] == 1)]) else None for w in range(5)],
                  "float": [CAT["disenos"].get(ext_id, {}).get("float_min"), CAT["disenos"].get(ext_id, {}).get("float_max")]}

EXTERIOR_INTRO["imgs"] = [next(iter(ext[(ext["wear_orden"] == w) & (ext["var"] == 0)]["img"]), "") for w in range(5)]
par = todo.pivot_table(index=["skin_id", "wear_orden"], columns="var", values="sell_price_usd", aggfunc="min")
VECES_ST = float((par[1] / par[0]).dropna().median())
VECES_SV = float((par[2] / par[0]).dropna().median())

# un mismo diseño en sus tres variantes, para mostrar StatTrak y Souvenir lado a lado
tres = par.dropna(subset=[0, 1, 2]).reset_index()
tres = tres[tres["skin_id"].map(dict(zip(dis["skin_id"], dis["arma"]))).isin(POPULARES + ["MP9", "P250", "Five-SeveN", "MAC-10", "FAMAS", "Galil AR"])]
if len(tres):
    t_ = tres.assign(rr=tres[1] / tres[0]).sort_values("rr").iloc[len(tres) // 2]
    VARIANTES_INTRO = {"d": i_dis[t_["skin_id"]], "w": int(t_["wear_orden"]),
                       "p": [round(float(t_[0]), 2), round(float(t_[1]), 2), round(float(t_[2]), 2)],
                       "img": next(iter(todo[(todo["skin_id"] == t_["skin_id"]) & (todo["wear_orden"] == t_["wear_orden"])]["img"]), "")}
else:
    VARIANTES_INTRO = None
ORIGEN_INTRO, usados = [], set()
for gi, gr in enumerate(ORIGENES):        # las cuatro etiquetas tal como las ve el modelo
    g = todo[todo["origen"] == gr]
    cols_g = g["collection"].value_counts()
    ej = next((c for c in cols_g.index if img_coleccion(c).startswith("img/") and img_coleccion(c) not in usados), cols_g.index[0])
    usados.add(img_coleccion(ej))
    ORIGEN_INTRO.append({"filas": len(g), "colecciones": int(g["collection"].nunique()),
                         "mediana": round(float(g["sell_price_usd"].median()), 2), "q": round(float(g["q"].median()), 0),
                         "ej_col": corto(ej), "ej_icono": img_coleccion(ej)})

FILA_INTRO = []
for k in (3, 15, 30, 50, 70, 85, 97):
    r = todo.iloc[(todo["q"] - k).abs().argsort().iloc[0]]
    FILA_INTRO.append(fila_ejemplo(r))

caja_demo = "The Kilowatt Collection" if "The Kilowatt Collection" in i_col else \
    todo[todo["origen"] == "caja"]["collection"].value_counts().index[0]
g = todo[(todo["collection"] == caja_demo) & (todo["var"] == 0)]
CAJA_INTRO = {"c": i_col[caja_demo], "caja": g["caja"].dropna().iloc[0] if g["caja"].notna().any() else "",
              "items": [{"d": i_dis[s], "r": int(gg["rarity_orden"].iloc[0]),
                         "p": round(float(gg.loc[gg["wear_orden"] == 2, "sell_price_usd"].min()
                                          if (gg["wear_orden"] == 2).any() else gg["sell_price_usd"].median()), 2)}
                        for s, gg in g.groupby("skin_id")]}
CAJA_INTRO["img"] = img_caja(CAJA_INTRO["caja"])

INTRO = {"variantes": VARIANTES_INTRO, "gancho": GANCHO, "rareza": RAREZA_INTRO, "exterior": EXTERIOR_INTRO, "veces_st": round(VECES_ST, 2),
         "veces_sv": round(VECES_SV, 2), "origen": ORIGEN_INTRO, "fila": FILA_INTRO, "caja": CAJA_INTRO}

# ================================================================ 12 · la defensa: ¿y si el modelo supiera qué es «drop nuevo»?
# Mismo procedimiento, con un corte que deja colecciones de la era Armory en el entrenamiento
# (prueba: colecciones desde 03/2025). Hiperparámetros: los que eligió la grilla de cada modelo
# con los pedazos temporales de ese entrenamiento (exp_modelos_armory.py, en la Entrega 4).
print("experimento: drop nuevo como origen propio…")
CORTE_EXP = pd.Timestamp("2025-03-01")
PARAMS_EXP = {
    "sin": {"Regresión lineal (Ridge)": Ridge(alpha=10),
            "Árbol de decisión": DecisionTreeRegressor(min_samples_leaf=40, random_state=SEMILLA),
            "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=SEMILLA, n_jobs=-1),
            "Gradient Boosting": HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=40,
                                                               random_state=SEMILLA)},
    "con": {"Regresión lineal (Ridge)": Ridge(alpha=10),
            "Árbol de decisión": DecisionTreeRegressor(min_samples_leaf=20, random_state=SEMILLA),
            "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=SEMILLA, n_jobs=-1),
            "Gradient Boosting": HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=40,
                                                               random_state=SEMILLA)},
}
fe_ = df["collection_release"].fillna(pd.Timestamp("2000-01-01"))
tr_e, te_e = df[fe_ < CORTE_EXP].reset_index(drop=True), df[fe_ >= CORTE_EXP].reset_index(drop=True)
M_e = np.sort(tr_e["sell_price_usd"].to_numpy(float))
ytr_e, yte_e = percentil_contra(tr_e["sell_price_usd"], M_e), percentil_contra(te_e["sell_price_usd"], M_e)
mapa_e = (te_e["origen"] == "drop de mapa").to_numpy()
DEFENSA = []
for version, modelos in PARAMS_EXP.items():
    Xa, Xb = tr_e[COLUMNAS_X].copy(), te_e[COLUMNAS_X].copy()
    if version == "con":
        Xa["origen"] = np.where(tr_e["grupo"] == "drop nuevo", "drop nuevo", tr_e["origen"])
        Xb["origen"] = np.where(te_e["grupo"] == "drop nuevo", "drop nuevo", te_e["origen"])
    for nombre, reg in modelos.items():
        p = armar_modelo(reg).fit(Xa, ytr_e).predict(Xb)
        DEFENSA.append({"modelo": nombre, "version": version, "r2": float(r2_score(yte_e, p)),
                        "sesgo_mapa": float(np.mean(p[mapa_e] - yte_e[mapa_e])),
                        "arriba": float(100 * np.mean(p > yte_e))})
p_tab = MediaPorGrupo(("rarity", "wear")).fit(tr_e, ytr_e).predict(te_e)
DEFENSA_TABLA = float(r2_score(yte_e, p_tab))
DEF_INFO = {"train": len(tr_e), "test": len(te_e), "col_test": int(te_e["collection"].nunique()),
            "nuevo_train": int((tr_e["grupo"] == "drop nuevo").sum()), "nuevo_test": int((te_e["grupo"] == "drop nuevo").sum())}
print("  ", [(d["modelo"][:6], d["version"], round(d["r2"], 3)) for d in DEFENSA], "tabla", round(DEFENSA_TABLA, 3))

# ================================================================ 13 · gráficos (Altair)
COL_RAREZA = ["#9fb3cc", "#5e98d9", "#4b69ff", "#8847ff", "#d32ce6", "#eb4b4b"]   # los colores de CS2
esc_rareza = alt.Scale(domain=RAREZAS, range=COL_RAREZA)
skins = alt.NamedData("skins")
rareza_nombre = "['Consumer Grade','Industrial Grade','Mil-Spec Grade','Restricted','Classified','Covert'][datum.r]"
GRIS, AMBAR, CORAL, AZUL = "#8a93a0", "#f0a93b", "#ff5f3d", "#4b69ff"

# 13.1 los datos: la mitad barata y la mitad cara, en escala logarítmica
mediana_precio = float(np.median(df["sell_price_usd"]))
p90_precio = float(np.quantile(df["sell_price_usd"], 0.9))
lp90 = float(np.log10(p90_precio))
lmed = float(np.log10(mediana_precio))
n_baratos = int((df["sell_price_usd"] <= mediana_precio).sum())
# Los tramos se cuentan acá: más simple y sin depender del orden de las transformaciones de Vega-Lite.
lp_ = np.log10(df["sell_price_usd"].to_numpy(float))
bordes = np.round(np.arange(-1.6, 3.41, 0.1), 2)
cuenta, _ = np.histogram(lp_, bins=bordes)
tramos_hist = [{"a": float(a), "b": float(b), "n": int(c),
                "mitad": "barata" if (a + b) / 2 <= lmed else ("carisima" if (a + b) / 2 >= lp90 else "cara"),
                "desde": f"USD {n(10 ** a, 2)}", "hasta": f"USD {n(10 ** b, 2)}"}
               for a, b, c in zip(bordes[:-1], bordes[1:], cuenta) if c > 0]
X_LP = alt.X("a:Q", scale=alt.Scale(domain=[-1.6, 3.4], nice=False, zero=False),
             axis=alt.Axis(values=[-1, 0, 1, 2, 3], labelExpr="'USD ' + format(pow(10, datum.value), ',~r')"),
             title="Precio en Steam (cada marca vale 10 veces la anterior)")
hist = (alt.Chart(alt.Data(values=tramos_hist)).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(x=X_LP,
                x2="b:Q", y=alt.Y("n:Q", title="Cantidad de skins"), y2=alt.datum(0),
                color=alt.Color("mitad:N", scale=alt.Scale(domain=["barata", "cara", "carisima"], range=[AMBAR, "#5b6470", CORAL]), legend=None),
                tooltip=[alt.Tooltip("desde:N", title="Desde"), alt.Tooltip("hasta:N", title="Hasta"),
                         alt.Tooltip("n:Q", title="Skins")]))
regla = (alt.Chart(alt.Data(values=[{"a": lmed}, {"a": lp90}])).mark_rule(strokeDash=[4, 3], strokeWidth=2, color="#e6e8eb")
         .encode(x=X_LP))
def texto(valores, **props):
    return alt.Chart(alt.Data(values=valores)).mark_text(**props)


txt_mitades = (texto([{"a": lmed - 0.08, "t": f"← la mitad cuesta menos de USD {n(mediana_precio, 2)}"}],
                     align="right", fontWeight=600, fontSize=12.5, color="#e6e8eb").encode(x=X_LP, y=alt.value(14), text="t:N")
               + texto([{"a": lp90 + 0.06, "t": f"1 de cada 10 cuesta más de USD {n(p90_precio, 0)} →"}],
                       align="left", fontWeight=600, fontSize=12.5, color="#ffb4a3").encode(x=X_LP, y=alt.value(34), text="t:N"))
extremos = df.loc[[df["sell_price_usd"].idxmin(), df["sell_price_usd"].idxmax()]]
ext_vals = [{"a": float(np.log10(r.sell_price_usd)), "t": f"{'El más barato' if i == 0 else 'El más caro'}: USD {n(r.sell_price_usd, 2)}"}
            for i, r in enumerate(extremos.itertuples())]
marcas_ext = (alt.Chart(alt.Data(values=ext_vals)).mark_point(shape="triangle-down", size=90, filled=True, color=CORAL)
              .encode(x=X_LP, y=alt.value(52)))
txt_ext = (texto([ext_vals[0]], fontSize=11.5, dy=-14, dx=-6, color=CORAL, align="left").encode(x=X_LP, y=alt.value(52), text="t:N")
           + texto([ext_vals[1]], fontSize=11.5, dy=-14, dx=6, color=CORAL, align="right").encode(x=X_LP, y=alt.value(52), text="t:N"))
graf_datos = (hist + regla + txt_mitades).properties(width="container", height=260)

# 13.2 lo que encontramos, en tres pasos
def medianas(datos, clave):
    m = datos.groupby("rarity_orden")["q"].agg(["median", "size"]).reset_index()
    m = m[m["size"] >= 20]
    m["rareza"] = m["rarity_orden"].map(dict(enumerate(RAREZAS)))
    m["k"] = clave
    return m


series = [medianas(df, "todos")]
for o in ["caja", "drop de mapa", "paquete souvenir", "sin identificar"]:
    series.append(medianas(df[df["origen"] == o], o))
series.append(medianas(df[df["grupo"] == "drop de mapa clásico"], "clasico"))
series.append(medianas(df[df["grupo"] == "drop nuevo"], "nuevo"))
med_series = pd.concat(series, ignore_index=True)
MED = {(k, int(r)): float(m) for k, r, m in zip(med_series["k"], med_series["rarity_orden"], med_series["median"])}
NOMBRE_SERIE = {"todos": "Todos juntos", "caja": "Caja", "drop de mapa": "Drop de mapa",
                "paquete souvenir": "Paquete de torneo", "sin identificar": "Souvenir sin paquete",
                "clasico": NOMBRE_GRUPO[1], "nuevo": NOMBRE_GRUPO[2]}
COLOR_SERIE = {"todos": "#c3c9d1", "caja": COL_ORIGEN["caja"], "drop de mapa": COL_ORIGEN["drop de mapa"],
               "paquete souvenir": COL_ORIGEN["paquete souvenir"], "sin identificar": COL_ORIGEN["sin identificar"],
               "clasico": COL_GRUPO[1], "nuevo": COL_GRUPO[2]}
med_series["serie"] = med_series["k"].map(NOMBRE_SERIE)
p_paso = alt.param(name="paso", value=1)
OPACIDAD = ("(datum.k == 'clasico' || datum.k == 'nuevo') ? 0 : paso == 1 ? (datum.k == 'todos' ? 1 : 0)"
            " : (datum.k == 'todos' ? 0.22 : 1)")
x_r = alt.X("rareza:N", sort=RAREZAS, title=None, axis=alt.Axis(labelAngle=0, labelExpr="split(datum.label, ' ')[0]"))
y_r = alt.Y("median:Q", scale=alt.Scale(domain=[0, 100]), title="Puesto típico en el mercado (0 = el más barato)")
esc_serie = alt.Scale(domain=list(NOMBRE_SERIE.values()), range=[COLOR_SERIE[k] for k in NOMBRE_SERIE])
b_h = (alt.Chart(alt.Data(values=med_series.to_dict(orient="records")))
       .transform_calculate(op=OPACIDAD).transform_filter("datum.op > 0"))
tip_h = [alt.Tooltip("serie:N", title="Grupo"), alt.Tooltip("rareza:N", title="Rareza"),
         alt.Tooltip("median:Q", title="Puesto típico", format=".0f"), alt.Tooltip("size:Q", title="Skins")]
lin_h = b_h.mark_line(strokeWidth=3).encode(x=x_r, y=y_r, color=alt.Color("serie:N", scale=esc_serie, legend=None),
                                             opacity=alt.Opacity("op:Q", scale=None), detail="k:N",
                                             strokeDash=alt.condition("datum.k == 'todos'", alt.value([6, 4]), alt.value([1, 0])))
pts_h = b_h.mark_point(filled=True, size=80).encode(x=x_r, y=y_r, color=alt.Color("serie:N", scale=esc_serie, legend=None),
                                                     opacity=alt.Opacity("op:Q", scale=None), tooltip=tip_h)
NOTAS_H = [
    {"paso": 1, "rareza": RAREZAS[2], "median": MED[("todos", 2)], "t": "¿Más raro y más barato?", "dy": 30},
    {"paso": 2, "rareza": RAREZAS[1], "median": 93, "t": "Separados por cómo se consiguen, todos suben", "dy": 0},
]
anillo = (alt.Chart(alt.Data(values=[NOTAS_H[0]])).transform_filter("paso == 1")
          .mark_point(size=900, strokeWidth=2.5, color=CORAL).encode(x=x_r, y="median:Q"))
notas_h = alt.layer(*[texto([nota], fontWeight=600, fontSize=13, color="#e6e8eb", dy=nota["dy"])
                       .transform_filter(f"paso == {nota['paso']}").encode(x=x_r, y="median:Q", text="t:N")
                       for nota in NOTAS_H])
graf_hallazgo = (lin_h + pts_h + anillo + notas_h).add_params(p_paso).properties(width="container", height=340)

# 13.2b el drop de mapa partido por época (sección D del modelo)
epoca = med_series[med_series["k"].isin(["clasico", "nuevo"])].copy()
brecha = []
for r_i, r in enumerate(RAREZAS):
    if ("clasico", r_i) in MED and ("nuevo", r_i) in MED:
        a_, b_ = MED[("clasico", r_i)], MED[("nuevo", r_i)]
        brecha.append({"rareza": r, "a": a_, "b": b_, "m": (a_ + b_) / 2, "t": f"−{n(a_ - b_)} puestos"})
esc_epoca = alt.Scale(domain=[NOMBRE_GRUPO[1], NOMBRE_GRUPO[2]], range=[COL_GRUPO[1], COL_GRUPO[2]])
b_ep = alt.Chart(alt.Data(values=epoca.to_dict(orient="records")))
graf_epoca = (
    alt.Chart(alt.Data(values=brecha)).mark_rule(strokeWidth=10, opacity=0.18, color=CORAL).encode(x=x_r, y="a:Q", y2="b:Q")
    + b_ep.mark_line(strokeWidth=3.5).encode(x=x_r, y=y_r, color=alt.Color("serie:N", scale=esc_epoca,
                                             legend=alt.Legend(orient="top", title=None)), detail="k:N")
    + b_ep.mark_point(filled=True, size=110).encode(x=x_r, y=y_r, color=alt.Color("serie:N", scale=esc_epoca), tooltip=tip_h)
    + alt.Chart(alt.Data(values=brecha)).mark_text(dx=12, align="left", fontSize=11.5, fontWeight=600, color="#ffb4a3")
    .encode(x=x_r, y="m:Q", text="t:N")
).properties(width="container", height=320)

# 13.3 cómo lo probamos: las colecciones en el tiempo, de qué lado quedó cada una
part = pd.DataFrame([c for c in info_col if c["f"]])
part["lado_txt"] = np.where(part["lado"] == "prueba", "Prueba (nunca las vio)", "Entrenamiento (las vio)")
inicio_test = test["collection_release"].min()
graf_particion = (
    alt.Chart(alt.Data(values=part.to_dict(orient="records"))).mark_circle(opacity=0.85, stroke="#0e1014", strokeWidth=0.6)
    .encode(x=alt.X("f:T", title=None, axis=alt.Axis(format="%Y", tickCount=12), scale=alt.Scale(padding=14)),
            y=alt.Y("origen:N", sort=list(ORIGEN_TXT.values()), title=None),
            size=alt.Size("filas:Q", scale=alt.Scale(range=[30, 520]), legend=None),
            color=alt.Color("lado_txt:N", scale=alt.Scale(domain=["Entrenamiento (las vio)",
                                                                  "Prueba (nunca las vio)"], range=["#6f7b8a", AMBAR]),
                            legend=alt.Legend(orient="bottom", title=None, columns=1)),
            tooltip=[alt.Tooltip("nombre:N", title="Colección"), alt.Tooltip("f:T", title="Lanzada", format="%m/%Y"),
                     alt.Tooltip("tipo:N", title="Cómo se consigue"), alt.Tooltip("filas:Q", title="Skins")])
    + alt.Chart(alt.Data(values=[{"f": f"{inicio_test:%Y-%m-%d}", "t": "Desde acá, el modelo no vio nada"}]))
    .mark_rule(strokeDash=[4, 3], color="#c3c9d1").encode(x="f:T")
    + texto([{"f": f"{inicio_test:%Y-%m-%d}", "t": "Desde acá, el modelo no vio nada →"}], align="right", dx=-6,
            fontSize=11.5, fontWeight=600, color=AMBAR).encode(x="f:T", y=alt.value(-8), text="t:N")
).properties(width="container", height=250)

# 13.4 el resultado: validación contra prueba (pesas), con el elegido resaltado
prueba_largo = []
for r in PRUEBA:
    prueba_largo += [{"modelo": r["modelo"], "momento": "Validación (colecciones conocidas)", "r2": r["validacion"], "orden": 0,
                      "elegido": r["modelo"] == ELEGIDO},
                     {"modelo": r["modelo"], "momento": "Prueba (colecciones nuevas)", "r2": r["prueba"], "orden": 1,
                      "elegido": r["modelo"] == ELEGIDO}]
orden_mod = [r["modelo"] for r in PRUEBA][::-1]
b_m = alt.Chart(alt.Data(values=prueba_largo))
lineas_m = b_m.mark_line(strokeWidth=3).encode(
    x=alt.X("r2:Q", scale=alt.Scale(domain=[-0.2, 0.7]), title="R²: cuánto del orden de precios explica (0 = nada, 1 = todo)"),
    y=alt.Y("modelo:N", sort=orden_mod, title=None), detail="modelo:N", order="orden:Q",
    color=alt.condition("datum.elegido", alt.value(CORAL), alt.value("#56606c")))
pts_m = b_m.mark_point(size=170, filled=True).encode(
    x="r2:Q", y=alt.Y("modelo:N", sort=orden_mod),
    color=alt.Color("momento:N", scale=alt.Scale(domain=["Validación (colecciones conocidas)", "Prueba (colecciones nuevas)"],
                                                 range=[AZUL, AMBAR]), legend=alt.Legend(orient="bottom", title=None)),
    tooltip=[alt.Tooltip("modelo:N", title="Modelo"), alt.Tooltip("momento:N", title="Momento"),
             alt.Tooltip("r2:Q", title="R²", format=".3f")])
txt_m = b_m.mark_text(dy=-15, fontWeight=600, color="#e6e8eb").encode(x="r2:Q", y=alt.Y("modelo:N", sort=orden_mod),
                                                                       text=alt.Text("r2:Q", format=".2f"))
cero = alt.Chart(alt.Data(values=[{"x": 0}])).mark_rule(color=GRIS, strokeDash=[2, 3]).encode(x="x:Q")
graf_modelo = (cero + lineas_m + pts_m + txt_m).properties(width="container", height=200)

# 13.5 skin por skin: real contra predicho, color = rareza (como en el juego)
p_vista = alt.param(name="vista", value=0)
p_res = alt.param(name="resaltar", value=0)
p_punto = alt.param(name="punto", value=-1)
nube = (alt.Chart(skins).transform_window(idx="row_number()").transform_filter("datum.l == vista")
        .transform_calculate(rareza=rareza_nombre, foco="resaltar == 0 || datum.g == 2")
        .mark_circle(stroke=None)
        .encode(x=alt.X("y:Q", scale=alt.Scale(domain=[0, 100]), title="Puesto real en el mercado"),
                y=alt.Y("yh:Q", scale=alt.Scale(domain=[0, 100]), title="Puesto que dijo el modelo"),
                color=alt.Color("rareza:N", scale=esc_rareza, legend=alt.Legend(orient="bottom", title=None, labelExpr="split(datum.label, ' ')[0]")),
                opacity=alt.condition("datum.foco", alt.value(0.7), alt.value(0.06)),
                size=alt.condition("datum.idx - 1 == punto", alt.value(260), alt.value(22)),
                tooltip=alt.Tooltip("idx:Q")))
zona_sobre = (alt.Chart(alt.Data(values=[{"x": 0, "y": 0, "y2": 100}, {"x": 100, "y": 100, "y2": 100}]))
              .mark_area(color=CORAL, opacity=0.07).encode(x="x:Q", y="y:Q", y2="y2:Q"))
diag = (alt.Chart(alt.Data(values=[{"x": 0, "y": 0}, {"x": 100, "y": 100}]))
        .mark_line(color="#e6e8eb", strokeDash=[5, 4], strokeWidth=1.5).encode(x="x:Q", y="y:Q"))
txt_zona = (texto([{"x": 3, "y": 96, "t": "Arriba de la línea: el modelo lo pone más caro de lo que es"}], fontSize=11.5, color="#ffb4a3",
                  align="left").encode(x="x:Q", y="y:Q", text="t:N")
            + texto([{"x": 97, "y": 4, "t": "Abajo: lo pone más barato"}], fontSize=11.5, color="#c3c9d1",
                    align="right").encode(x="x:Q", y="y:Q", text="t:N"))
graf_nube = (zona_sobre + diag + nube + txt_zona).add_params(p_vista, p_res, p_punto).properties(width="container", height=420)

# 13.6 la defensa: avisarle al modelo qué es «drop nuevo»
def_df = pd.DataFrame(DEFENSA)
def_df["version_txt"] = def_df["version"].map({"sin": "Hoy: sin saber qué es el Armory",
                                              "con": "Conociendo el Armory"})
orden_def = list(PARAMS_EXP["sin"].keys())
b_d = alt.Chart(alt.Data(values=def_df.to_dict(orient="records")))
esc_ver = alt.Scale(domain=["Hoy: sin saber qué es el Armory", "Conociendo el Armory"],
                    range=["#5b6470", CORAL])
graf_defensa = (
    b_d.mark_bar(height=15, cornerRadiusEnd=3).encode(
        x=alt.X("r2:Q", scale=alt.Scale(domain=[0, 1]), title="R²: cuánto del orden de precios acierta (1 = todo)"),
        y=alt.Y("modelo:N", sort=orden_def, title=None), yOffset=alt.YOffset("version_txt:N", sort=esc_ver.domain),
        color=alt.Color("version_txt:N", scale=esc_ver, legend=alt.Legend(orient="bottom", title=None, columns=1)),
        tooltip=[alt.Tooltip("modelo:N", title="Modelo"), alt.Tooltip("version_txt:N", title="Versión"),
                 alt.Tooltip("r2:Q", title="R²", format=".3f"), alt.Tooltip("sesgo_mapa:Q", title="Sesgo en drop", format="+.1f")])
    + b_d.mark_text(align="left", dx=5, fontSize=11.5, fontWeight=600, color="#e6e8eb").encode(
        x="r2:Q", y=alt.Y("modelo:N", sort=orden_def), yOffset=alt.YOffset("version_txt:N", sort=esc_ver.domain),
        text=alt.Text("r2:Q", format=".2f"))
    + alt.Chart(alt.Data(values=[{"x": DEFENSA_TABLA, "t": f"Tabla de rareza y exterior: {n(DEFENSA_TABLA, 2)}"}]))
    .mark_rule(strokeDash=[4, 3], color=GRIS).encode(x="x:Q")
).properties(width="container", height=250)

# 13.7 cómo lo sabemos del modelo: comparación con media y desvío, e importancia
comp = [{"modelo": r["modelo"], "r2": r["r2"], "lo": r["r2"] - r["desvio"], "hi": r["r2"] + r["desvio"],
         "tipo": "piso" if r in pisos else ("elegido" if r["modelo"] == ELEGIDO else "modelo")}
        for r in pisos + comparacion]
orden_c = [c["modelo"] for c in sorted(comp, key=lambda c: -c["r2"])]
b_c = alt.Chart(alt.Data(values=comp))
esc_tipo = alt.Scale(domain=["piso", "modelo", "elegido"], range=[GRIS, AZUL, CORAL])
graf_comp = (b_c.mark_rule(strokeWidth=3, opacity=0.5).encode(
                 x=alt.X("lo:Q", title="R² en validación (media ± desvío entre pedazos)"), x2="hi:Q",
                 y=alt.Y("modelo:N", sort=orden_c, title=None), color=alt.Color("tipo:N", scale=esc_tipo, legend=None))
             + b_c.mark_point(filled=True, size=110).encode(
                 x="r2:Q", y=alt.Y("modelo:N", sort=orden_c), color=alt.Color("tipo:N", scale=esc_tipo, legend=None),
                 tooltip=[alt.Tooltip("modelo:N", title="Modelo"), alt.Tooltip("r2:Q", format=".3f", title="R²"),
                          alt.Tooltip("lo:Q", format=".3f", title="− desvío"), alt.Tooltip("hi:Q", format=".3f", title="+ desvío")])
             ).properties(width="container", height=230)
graf_imp = (alt.Chart(alt.Data(values=IMPORTANCIA)).mark_bar(cornerRadiusEnd=3, height=16)
            .encode(x=alt.X("baja:Q", title="Cuánto empeora el R² si se mezcla esa columna al azar"),
                    y=alt.Y("columna:N", sort="-x", title=None),
                    color=alt.condition(alt.FieldOneOfPredicate("columna", ["Rareza", "Origen"]), alt.value(CORAL), alt.value("#5b6470")),
                    tooltip=[alt.Tooltip("columna:N", title="Columna"), alt.Tooltip("baja:Q", format=".3f", title="Baja")])
            ).properties(width="container", height=210)

# 13.8 explorá: dónde queda la colección elegida dentro de todo el mercado
p_sel = alt.param(name="sel", value=-1)
p_arma = alt.param(name="f_arma", value=-1)
p_rar = alt.param(name="f_rareza", value=-1)
base_pan = (alt.Chart(skins).transform_calculate(rareza=rareza_nombre, j="random()",
                                                 ok="(f_arma < 0 || datum.a == f_arma) && (f_rareza < 0 || datum.r == f_rareza)"))
y_pan = alt.Y("rareza:N", sort=RAREZAS[::-1], title=None, axis=alt.Axis(labelExpr="split(datum.label, ' ')[0]"))
x_pan = alt.X("p:Q", scale=alt.Scale(type="log", domain=[0.02, 2000], nice=False), title="Precio en Steam (escala de 10 en 10)",
              axis=alt.Axis(values=[0.03, 0.1, 1, 10, 100, 1000], labelExpr="'USD ' + format(datum.value, ',~r')"))
fondo_pan = (base_pan.transform_filter("datum.c != sel").mark_circle(size=10, color="#5b6470", stroke=None)
             .encode(x=x_pan, y=y_pan, yOffset=alt.YOffset("j:Q", scale=alt.Scale(domain=[0, 1])),
                     opacity=alt.condition("datum.ok", alt.value(0.28), alt.value(0.05))))
elegidos = (base_pan.transform_filter("datum.c == sel").mark_circle(size=70, stroke="#0e1014", strokeWidth=0.8)
            .encode(x=x_pan, y=y_pan, yOffset=alt.YOffset("j:Q", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("rareza:N", scale=esc_rareza, legend=None),
                    opacity=alt.condition("datum.ok", alt.value(0.95), alt.value(0.15)),
                    tooltip=[alt.Tooltip("p:Q", title="Precio (USD)", format=",.2f")]))
graf_explora = (fondo_pan + elegidos).add_params(p_sel, p_arma, p_rar).properties(width="container", height=190)

# 13.9 probá el modelo: dónde cae el skin fabricado en la escalera del mercado
p_pred = alt.param(name="pred", value=50)
p_lo = alt.param(name="lo", value=35)
p_hi = alt.param(name="hi", value=65)
fondo = (alt.Chart(skins).transform_calculate(rareza=rareza_nombre)
         .mark_tick(orient="vertical", thickness=1, opacity=0.35)
         .encode(x=alt.X("q:Q", scale=alt.Scale(domain=[0, 100]), title="Puesto en el mercado (0 = el más barato, 100 = el más caro)"),
                 color=alt.Color("rareza:N", scale=esc_rareza, legend=None)))
uno = alt.Chart(alt.Data(values=[{}]))
banda = uno.transform_calculate(a="lo", b="hi").mark_rect(color=AMBAR, opacity=0.18).encode(x="a:Q", x2="b:Q")
marca = uno.transform_calculate(a="pred").mark_rule(color=AMBAR, strokeWidth=3).encode(x="a:Q")
punta = uno.transform_calculate(a="pred").mark_point(shape="triangle-down", size=260, filled=True, color=AMBAR) \
    .encode(x="a:Q", y=alt.value(-6))
graf_proba = (fondo + banda + marca + punta).add_params(p_pred, p_lo, p_hi).properties(width="container", height=90)

ESPECS = {k: v.to_dict() for k, v in {
    "datos": graf_datos, "hallazgo": graf_hallazgo, "particion": graf_particion, "modelo": graf_modelo,
    "nube": graf_nube, "defensa": graf_defensa, "epoca": graf_epoca, "comparacion": graf_comp, "importancia": graf_imp,
    "explora": graf_explora, "proba": graf_proba}.items()}

# ================================================================ 14 · textos y página
fila_tabla = lambda celdas: "<tr>" + "".join(f"<td>{c}</td>" for c in celdas) + "</tr>"
T = {}
T["SNAPSHOT"] = date.fromisoformat(SNAPSHOT).strftime("%d/%m/%Y")
T["FILAS"] = n(len(df))
T["COLECCIONES"] = n(df["collection"].nunique())
T["SKINS_BASE"] = n(df["skin_id"].nunique())
T["ARMAS_N"] = n(len(ARMAS))
T["PRECIO_MIN"] = n(df["sell_price_usd"].min(), 2)
T["PRECIO_MAX"] = n(df["sell_price_usd"].max(), 0)
T["PRECIO_MED"] = n(mediana_precio, 2)
T["PRECIO_P90"] = "USD " + n(p90_precio, 0)
T["VECES_RANGO"] = n(df["sell_price_usd"].max() / df["sell_price_usd"].min(), 0)
T["COBERTURA"] = CALIDAD.get("cobertura", "—").replace(".", ",")
T["MERCADO_ARMAS"] = n(CALIDAD.get("items_de_arma_en_el_mercado", EMBUDO.get("steam_items_de_arma", 0)))
T["DATOS_FRESCOS"] = "sí" if CALIDAD.get("datos_frescos") else "no"
e = EMBUDO
T["EMBUDO_STEAM"] = n(e.get("steam_items_de_arma", 0))
T["EMBUDO_LEIDOS"] = n(e.get("steam_items_leidos", 0))
T["EMBUDO_PAGINAS"] = n(e.get("steam_paginas", 0))
T["EMBUDO_DUP"] = n(e.get("steam_items_leidos", 0) - e.get("steam_sin_duplicados", 0))
T["EMBUDO_CATALOGO"] = n(e.get("catalogo_armas", 0))
T["PEDAZOS"] = "".join(fila_tabla((p["pedazo"], n(p["entrena"]), n(p["valida"]), f"{p['desde']} a {p['hasta']}"))
                       for p in tabla_pedazos)
T["N_TRAIN"], T["N_TEST"] = n(len(train)), n(len(test))
T["COL_TRAIN"], T["COL_TEST"] = n(train["collection"].nunique()), n(test["collection"].nunique())
T["TEST_DESDE"] = f"{test['collection_release'].min():%m/%Y}"

T["H2_D"], T["H2_PARES"], T["H2_MED"] = n(H2["d"], 2), n(H2["pares"]), n(H2["mediana"], 0)
T["H3_RHO"], T["H3_SKINS"], T["H3_BAJA"] = n(H3["rho"], 2), n(H3["skins"]), n(H3["baja"])
T["H4_R"], T["H4_FILAS"] = n(H4["r"], 3), n(H4["filas"])
T["SPEARMAN_RAREZA"] = n(SPEARMAN_RAREZA, 2)
T["MED_CAJA_MIL"], T["MED_CAJA_COV"] = n(MED[("caja", 2)]), n(MED[("caja", 5)])
T["MED_TODOS_CONS"], T["MED_TODOS_MIL"] = n(MED[("todos", 0)]), n(MED[("todos", 2)])
T["MED_CLAS_MIL"], T["MED_NUEVO_MIL"] = n(MED.get(("clasico", 2), 0)), n(MED.get(("nuevo", 2), 0))
T["MED_CLAS_RES"], T["MED_NUEVO_RES"] = n(MED.get(("clasico", 3), 0)), n(MED.get(("nuevo", 3), 0))
nuevo = df[df["grupo"] == "drop nuevo"]
clasico = df[df["grupo"] == "drop de mapa clásico"]
T["N_NUEVO"], T["COL_NUEVO"] = n(len(nuevo)), n(nuevo["collection"].nunique())
T["USD_NUEVO"], T["USD_CLASICO"] = n(nuevo["sell_price_usd"].median(), 2), n(clasico["sell_price_usd"].median(), 2)
T["VECES_CLAS_NUEVO"] = n(clasico["sell_price_usd"].median() / nuevo["sell_price_usd"].median(), 0)

T["ELEGIDO"] = ELEGIDO
T["R2_CV"], T["DESVIO_CV"] = n(R2_CV, 2), n(DESVIO_CV, 2)
T["R2_CV3"], T["DESVIO_CV3"] = n(R2_CV, 3), n(DESVIO_CV, 3)
T["R2_CV_PCT"] = pct(100 * R2_CV)
T["R2_PISO"] = n(PISO["r2"], 2)
T["R2_TABLA"] = n(MEJOR_PISO["r2"], 2)
T["R2_TEST"] = n(R2_TEST, 2)
T["R2_TABLA_TEST"] = n(PRUEBA[1]["prueba"], 2)
T["R2_PISO_TEST"] = n(PRUEBA[0]["prueba"], 2)
T["MAE_CV"], T["MAE_TEST"] = n(MAE_CV, 1), n(MAE_TEST, 1)
T["SESGO_TEST1"] = n(SESGO_TEST, 1)
T["SESGO_PISO_TEST"] = n(PRUEBA[0]["sesgo_prueba"], 1)
T["COL_SESGO_POS"] = f"{COLECCIONES_SESGO_POS} de {test['collection'].nunique()}"
T["MAS_CARO"] = pct(USD["mas_caro"])
T["USD_MED"], T["USD_MEDIO"] = n(USD["abs_mediano"], 2), n(USD["abs_medio"], 2)
T["USD_VECES"], T["USD_X2"] = n(USD["veces_mediana"], 1), pct(USD["dentro_x2"])
T["MAPA_TR"] = n(next(c["mediana"] for c in corrimiento if c["origen"] == "drop de mapa" and c["lado"] == "entrenamiento"), 1)
T["MAPA_TE"] = n(next(c["mediana"] for c in corrimiento if c["origen"] == "drop de mapa" and c["lado"] == "prueba"), 1)
T["MAPA_SESGO"] = n(next(c["sesgo"] for c in corrimiento if c["origen"] == "drop de mapa" and c["lado"] == "prueba"), 1)
T["MEDIANA_TEST"] = n(test["percentil"].median(), 0)
T["DUELOS"] = "".join(fila_tabla((d["retador"], ("+" if d["dif"] >= 0 else "−") + n(abs(d["dif"]), 3),
                                  n(d["desvio_dif"], 3), f"{d['gana']} de {len(PEDAZOS)}",
                                  "<b>lo reemplaza</b>" if d["supera"] and d["retador"] == ELEGIDO else "no alcanza"))
                      for d in duelos)
T["COMPARACION"] = "".join(fila_tabla((r["modelo"], n(r["r2"], 3), n(r["desvio"], 3), n(r["mae"], 1),
                                       ("+" if r["sesgo"] >= 0 else "−") + n(abs(r["sesgo"]), 1)))
                           for r in sorted(pisos + comparacion, key=lambda r: -r["r2"]))
T["MEJORES"] = ", ".join(f"<code>{k}={v}</code>" for k, v in MEJORES.items())
T["N_GRILLA"] = str(N_GRILLA)
T["AJUSTE_DIF"], T["AJUSTE_DESV"] = n(AJUSTE["dif"], 3), n(AJUSTE["desvio_dif"], 3)
T["AJUSTE_LECTURA"] = ("sube por más que el desvío de la diferencia, así que vale la pena"
                       if AJUSTE["dif"] > AJUSTE["desvio_dif"] else "queda dentro del ruido")
T["TRAMOS"] = "".join(fila_tabla((str(t.tramo), n(t.filas), n(t.mae, 1), ("+" if t.sesgo >= 0 else "−") + n(abs(t.sesgo), 1),
                                  "USD " + n(t.real, 2), "USD " + n(t.pred, 2), "USD " + n(t.usd, 2)))
                      for t in por_tramo.itertuples())
T["PEORES"] = "".join(fila_tabla((r.market_hash_name, corto(r.collection), NOMBRE_GRUPO[GRUPOS.index(grupo_de(r.origen, fecha_col[r.collection]))],
                                  n(r.percentil, 0), n(r.predicho, 0), "USD " + n(r.sell_price_usd, 2),
                                  "USD " + n(r.precio_predicho, 2)))
                      for r in peores.itertuples())
T["IMP_1"], T["IMP_1_V"] = IMPORTANCIA[0]["columna"], n(IMPORTANCIA[0]["baja"], 2)
T["IMP_2"], T["IMP_2_V"] = IMPORTANCIA[1]["columna"], n(IMPORTANCIA[1]["baja"], 2)
T["R2_OOF"] = n(R2_OOF, 2)
T["VERIFICADO"] = VERIFICADO
T["SKLEARN"] = sklearn.__version__
T["N_GRILLA_APP"] = n(len(grilla))
T["VECES_ST"], T["VECES_SV"] = n(VECES_ST, 1), n(VECES_SV, 1)

defe = {(d["modelo"], d["version"]): d for d in DEFENSA}
gb_sin, gb_con = defe[("Gradient Boosting", "sin")], defe[("Gradient Boosting", "con")]
T["DEF_SIN"], T["DEF_CON"] = n(gb_sin["r2"], 2), n(gb_con["r2"], 2)
T["DEF_SESGO_SIN"], T["DEF_SESGO_CON"] = n(gb_sin["sesgo_mapa"], 1), n(gb_con["sesgo_mapa"], 1)
T["DEF_TABLA"] = n(DEFENSA_TABLA, 2)
T["DEF_TRAIN"], T["DEF_TEST"], T["DEF_COL_TEST"] = n(DEF_INFO["train"]), n(DEF_INFO["test"]), n(DEF_INFO["col_test"])
T["DEF_NUEVO_TRAIN"] = n(DEF_INFO["nuevo_train"])
T["DEF_TABLA_FILAS"] = "".join(fila_tabla((m, n(defe[(m, "sin")]["r2"], 2), n(defe[(m, "con")]["r2"], 2),
                                           ("+" if defe[(m, "sin")]["sesgo_mapa"] >= 0 else "−") + n(abs(defe[(m, "sin")]["sesgo_mapa"]), 1),
                                           ("+" if defe[(m, "con")]["sesgo_mapa"] >= 0 else "−") + n(abs(defe[(m, "con")]["sesgo_mapa"]), 1)))
                               for m in orden_def)

# números de la nube (los calcula también la página, pero así quedan en el texto)
for nombre_l, l_ in (("CONOC", 0), ("NUEVAS", 1)):
    s_ = todo[lado == l_]
    T[f"NUBE_{nombre_l}_R2"] = n(r2_score(s_["percentil"], s_["predicho"]), 2)
    T[f"NUBE_{nombre_l}_ARRIBA"] = pct(100 * (s_["predicho"] > s_["percentil"]).mean())
s_ = todo[(lado == 1) & (todo["grupo"] == "drop nuevo")]
T["NUBE_NUEVO_ARRIBA"] = pct(100 * (s_["predicho"] > s_["percentil"]).mean())
T["NUBE_NUEVO_SESGO"] = n((s_["predicho"] - s_["percentil"]).mean(), 1)

JS = {"IMGS": IMGS, "ARMA_IMG": ARMA_IMG, "COL_ORIGEN": list(COL_ORIGEN.values()), "SKINS": SKINS, "DISENOS": DISENOS, "COLS": info_col, "INTRO": INTRO, "ESPECS": ESPECS, "GRILLA": GRILLA,
      "VISTOS": VISTOS, "ARMAS": ARMAS, "ORIGENES": ORIGENES, "GRUPOS": NOMBRE_GRUPO, "COL_GRUPO": COL_GRUPO,
      "CUANTILES": CUANTILES_PRECIO, "MAE_CV": round(MAE_CV, 2), "SESGO_MAPA": round(float(T["MAPA_SESGO"].replace(",", ".")), 1),
      "COL_RAREZA": COL_RAREZA, "STEAM": STEAM}

html = Path(args.plantilla).read_text(encoding="utf-8")
for k, v in T.items():
    html = html.replace("{{" + k + "}}", str(v))
faltan = re.findall(r"\{\{([A-Z0-9_]+)\}\}", html)
assert not faltan, f"marcadores sin reemplazar: {faltan}"
datos_js = "const APP = " + json.dumps(JS, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";"
html = html.replace("/*__APP__*/", datos_js.replace("</", "<\\/"))
salida = Path(args.salida)
salida.parent.mkdir(parents=True, exist_ok=True)
salida.write_text(html, encoding="utf-8")
print(f"aplicación escrita en {salida} ({salida.stat().st_size / 1e6:.1f} MB)")
print(json.dumps({"r2_cv": R2_CV, "desvio": DESVIO_CV, "r2_test": R2_TEST, "mae_test": MAE_TEST, "sesgo_test": SESGO_TEST,
                  "elegido": ELEGIDO, "mejores": MEJORES, "usd": USD, "h2": H2, "h3": H3, "h4": H4,
                  "r2_oof": R2_OOF, "defensa": DEFENSA, "defensa_tabla": DEFENSA_TABLA}, default=str, indent=1))
