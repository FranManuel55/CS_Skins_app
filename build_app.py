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

# ================================================================ 10 · datos que viajan con la página
lado = np.r_[np.zeros(len(train), int), np.ones(len(test), int)]
todo = pd.concat([train, test], ignore_index=True)
todo["q"] = percentil_contra(todo["sell_price_usd"], MERCADO_TOTAL)
todo["ph"] = percentil_a_dolares(todo["predicho"].to_numpy())
VAR = np.select([todo["stattrak"] == 1, todo["souvenir"] == 1], ["StatTrak", "Souvenir"], "Normal")
SKINS = pd.DataFrame({
    "n": todo["market_hash_name"], "a": todo["weapon"], "r": todo["rarity_orden"].astype(int),
    "w": todo["wear_orden"].astype(int), "v": VAR, "o": todo["origen"],
    "c": todo["collection"].fillna("sin colección"),
    "f": todo["collection_release"].dt.strftime("%Y-%m-%d").fillna(""),
    "p": todo["sell_price_usd"].round(2), "q": todo["q"].round(1),
    "y": todo["percentil"].round(1), "yh": todo["predicho"].round(1), "ph": todo["ph"].round(2),
    "l": lado,
}).to_dict(orient="records")

# ================================================================ 11 · gráficos (Altair)
COL_RAREZA = ["#9fb3cc", "#5e98d9", "#4b69ff", "#8847ff", "#d32ce6", "#eb4b4b"]   # los colores de CS2
COL_ORIGEN = ["#2fae9f", "#e8962e", "#7e8ce0", "#9a8f86"]
esc_rareza = alt.Scale(domain=RAREZAS, range=COL_RAREZA)
esc_origen = alt.Scale(domain=ORIGENES, range=COL_ORIGEN)
skins = alt.NamedData("skins")
rareza_nombre = "['Consumer Grade','Industrial Grade','Mil-Spec Grade','Restricted','Classified','Covert'][datum.r]"

# 11.1 portada: el mercado en una tira
tira = (alt.Chart(skins).transform_calculate(rareza=rareza_nombre)
        .mark_tick(orient="vertical", thickness=1, opacity=0.55)
        .encode(x=alt.X("q:Q", scale=alt.Scale(domain=[0, 100]), title="Posición en el mercado del día (percentil)"),
                color=alt.Color("rareza:N", scale=esc_rareza, legend=alt.Legend(orient="bottom", title=None, columns=3,
                                                                                   symbolType="square")),
                tooltip=[alt.Tooltip("n:N", title="Skin"), alt.Tooltip("p:Q", title="USD", format=",.2f")])
        .properties(width="container", height=64))

# 11.2 los datos: precios en escala logarítmica
mediana_precio = float(np.median(df["sell_price_usd"]))
base_hist = (alt.Chart(skins).transform_calculate(lp="log(datum.p)/LN10")
             .transform_bin("lp_bin", "lp", bin=alt.Bin(step=0.1, extent=[-1.6, 3.4])))
hist = (base_hist.mark_bar(color="#c99a4b", cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(x=alt.X("lp_bin:Q", bin="binned", scale=alt.Scale(domain=[-1.6, 3.4]),
                        axis=alt.Axis(values=[-1, 0, 1, 2, 3], labelExpr="'USD ' + format(pow(10, datum.value), ',~r')"),
                        title="Precio de venta más bajo en Steam (escala logarítmica)"),
                x2="lp_bin_end:Q",
                y=alt.Y("count():Q", title="Skins"),
                tooltip=[alt.Tooltip("count():Q", title="Skins")]))
regla = (alt.Chart(alt.Data(values=[{"lp": float(np.log10(mediana_precio))}]))
         .mark_rule(strokeDash=[4, 3], strokeWidth=1.5, color="#e8962e").encode(x="lp:Q"))
etiqueta = (alt.Chart(alt.Data(values=[{"lp": float(np.log10(mediana_precio)),
                                        "t": f"La mitad cuesta menos de USD {mediana_precio:.2f}".replace(".", ",")}]))
            .mark_text(align="left", dx=6, dy=-6, fontWeight=600, color="#e8962e").encode(x="lp:Q", y=alt.value(12), text="t:N"))
graf_datos = (hist + regla + etiqueta).properties(width="container", height=240)

# 11.3 lo que encontramos: la rareza ordena el precio dentro de cada origen
med_ro = (df.groupby(["origen", "rarity_orden"])["q"].agg(["median", "size"]).reset_index()
            .query("size >= 20"))
med_ro["rareza"] = med_ro["rarity_orden"].map(dict(enumerate(RAREZAS)))
med_todos = df.groupby("rarity_orden")["q"].agg(["median", "size"]).reset_index()
med_todos["rareza"] = med_todos["rarity_orden"].map(dict(enumerate(RAREZAS)))
med_todos["origen"] = "todos juntos"
x_r = alt.X("rareza:N", sort=RAREZAS, title=None, axis=alt.Axis(labelAngle=0, labelExpr="split(datum.label, ' ')[0]"))
y_r = alt.Y("median:Q", scale=alt.Scale(domain=[0, 100]), title="Posición típica (percentil mediano)")
tip_r = [alt.Tooltip("origen:N"), alt.Tooltip("rareza:N"), alt.Tooltip("median:Q", title="Percentil mediano", format=".0f"),
         alt.Tooltip("size:Q", title="Skins")]
todos_l = (alt.Chart(alt.Data(values=med_todos.to_dict(orient="records")))
           .mark_line(strokeDash=[6, 4], strokeWidth=2.5, color="#8a93a0", point=alt.OverlayMarkDef(color="#8a93a0", size=50))
           .encode(x=x_r, y=y_r, tooltip=tip_r))
lineas_r = (alt.Chart(alt.Data(values=med_ro.to_dict(orient="records")))
            .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=70, filled=True))
            .encode(x=x_r, y=y_r, color=alt.Color("origen:N", scale=esc_origen, legend=alt.Legend(orient="bottom", title=None, columns=2)),
                    tooltip=tip_r))
fin = med_todos[med_todos["rarity_orden"] == 0]
txt_todos = (alt.Chart(alt.Data(values=[{"rareza": RAREZAS[0], "median": float(fin["median"].iloc[0]), "t": "Todos juntos"}]))
             .mark_text(dy=-12, color="#8a93a0", fontWeight=600).encode(x=x_r, y="median:Q", text="t:N"))
graf_hallazgo = (todos_l + lineas_r + txt_todos).properties(width="container", height=300)
MED_RO = {(o, int(r)): float(m) for o, r, m in zip(med_ro["origen"], med_ro["rarity_orden"], med_ro["median"])}

# 11.4 el modelo: validación contra prueba
prueba_largo = []
for r in PRUEBA:
    prueba_largo += [{"modelo": r["modelo"], "momento": "validación", "r2": r["validacion"], "orden": 0},
                     {"modelo": r["modelo"], "momento": "prueba (colecciones nuevas)", "r2": r["prueba"], "orden": 1}]
orden_mod = [r["modelo"] for r in PRUEBA][::-1]
b_m = alt.Chart(alt.Data(values=prueba_largo))
lineas_m = b_m.mark_line(color="#8a93a0", strokeWidth=2).encode(
    x=alt.X("r2:Q", scale=alt.Scale(domain=[-0.2, 0.7]), title="R²  (0 = predecir el promedio, 1 = perfecto)"),
    y=alt.Y("modelo:N", sort=orden_mod, title=None), detail="modelo:N", order="orden:Q")
pts_m = b_m.mark_point(size=150, filled=True, strokeWidth=2).encode(
    x="r2:Q", y=alt.Y("modelo:N", sort=orden_mod),
    color=alt.Color("momento:N", scale=alt.Scale(domain=["validación", "prueba (colecciones nuevas)"], range=["#4b69ff", "#e8962e"]),
                    legend=alt.Legend(orient="bottom", title=None)),
    tooltip=[alt.Tooltip("modelo:N"), alt.Tooltip("momento:N"), alt.Tooltip("r2:Q", title="R²", format=".3f")])
txt_m = b_m.mark_text(dy=-14, fontWeight=600).encode(x="r2:Q", y=alt.Y("modelo:N", sort=orden_mod),
                                                     text=alt.Text("r2:Q", format=".2f"))
cero = alt.Chart(alt.Data(values=[{"x": 0}])).mark_rule(color="#8a93a0", strokeDash=[2, 3]).encode(x="x:Q")
graf_modelo = (cero + lineas_m + pts_m + txt_m).properties(width="container", height=210)

# 11.5 el Armory: percentil mediano por origen, entrenamiento contra prueba
b_a = alt.Chart(alt.Data(values=corrimiento))
lin_a = b_a.mark_line(strokeWidth=3, opacity=0.6).encode(
    x=alt.X("mediana:Q", scale=alt.Scale(domain=[0, 100]), title="Percentil mediano"),
    y=alt.Y("origen:N", sort=ORIGENES, title=None), color=alt.Color("origen:N", scale=esc_origen, legend=None),
    detail="origen:N")
pts_a = b_a.mark_point(size=140, strokeWidth=2.5).encode(
    x="mediana:Q", y=alt.Y("origen:N", sort=ORIGENES), color=alt.Color("origen:N", scale=esc_origen, legend=None),
    shape=alt.Shape("lado:N", scale=alt.Scale(domain=["entrenamiento", "prueba"], range=["circle", "triangle-left"]),
                    legend=alt.Legend(orient="bottom", title=None)),
    fill=alt.condition("datum.lado == 'entrenamiento'", alt.Color("origen:N", scale=esc_origen), alt.value("transparent")),
    tooltip=[alt.Tooltip("origen:N"), alt.Tooltip("lado:N"), alt.Tooltip("mediana:Q", format=".1f", title="Percentil mediano"),
             alt.Tooltip("filas:Q"), alt.Tooltip("sesgo:Q", format="+.1f", title="Sesgo del modelo")])
graf_armory = (lin_a + pts_a).properties(width="container", height=190)

# 11.6 cómo lo sabemos del modelo: la comparación con media y desvío
comp = [{"modelo": r["modelo"], "r2": r["r2"], "lo": r["r2"] - r["desvio"], "hi": r["r2"] + r["desvio"],
         "tipo": "piso" if r in pisos else ("elegido" if r["modelo"] == ELEGIDO else "modelo")}
        for r in pisos + comparacion]
orden_c = [c["modelo"] for c in sorted(comp, key=lambda c: -c["r2"])]
b_c = alt.Chart(alt.Data(values=comp))
graf_comp = (b_c.mark_rule(strokeWidth=3, opacity=0.5).encode(
                 x=alt.X("lo:Q", title="R² en validación (media ± desvío entre pedazos)"), x2="hi:Q",
                 y=alt.Y("modelo:N", sort=orden_c, title=None),
                 color=alt.Color("tipo:N", scale=alt.Scale(domain=["piso", "modelo", "elegido"],
                                                           range=["#8a93a0", "#4b69ff", "#e8962e"]), legend=None))
             + b_c.mark_point(filled=True, size=110).encode(
                 x="r2:Q", y=alt.Y("modelo:N", sort=orden_c),
                 color=alt.Color("tipo:N", scale=alt.Scale(domain=["piso", "modelo", "elegido"],
                                                           range=["#8a93a0", "#4b69ff", "#e8962e"]), legend=None),
                 tooltip=[alt.Tooltip("modelo:N"), alt.Tooltip("r2:Q", format=".3f", title="R²"),
                          alt.Tooltip("lo:Q", format=".3f", title="− desvío"), alt.Tooltip("hi:Q", format=".3f", title="+ desvío")])
             ).properties(width="container", height=230)

graf_imp = (alt.Chart(alt.Data(values=IMPORTANCIA)).mark_bar(color="#4b69ff", cornerRadiusEnd=3, height=16)
            .encode(x=alt.X("baja:Q", title="Cuánto baja el R² en la prueba al desordenar la columna"),
                    y=alt.Y("columna:N", sort="-x", title=None),
                    tooltip=[alt.Tooltip("columna:N"), alt.Tooltip("baja:Q", format=".3f"), alt.Tooltip("desvio:Q", format=".3f")])
            ).properties(width="container", height=210)

# 11.7 explorá: real contra predicho, filtrado por controles y por una época
p_origen = alt.param(name="f_origen", value="todos")
p_rareza = alt.param(name="f_rareza", value=-1)
p_arma = alt.param(name="f_arma", value="todas")
p_lado = alt.param(name="f_lado", value=-1)
epoca = alt.selection_interval(name="epoca", encodings=["x"])
filtro = ("(f_origen == 'todos' || datum.o == f_origen) && (f_rareza < 0 || datum.r == f_rareza)"
          " && (f_arma == 'todas' || datum.a == f_arma) && (f_lado < 0 || datum.l == f_lado)")
base_e = alt.Chart(skins).transform_calculate(fecha="toDate(datum.f)")
nube = (base_e.transform_filter(filtro).transform_filter(epoca)
        .mark_circle(size=22, opacity=0.6, stroke=None)
        .encode(x=alt.X("y:Q", scale=alt.Scale(domain=[0, 100]), title="Percentil real"),
                y=alt.Y("yh:Q", scale=alt.Scale(domain=[0, 100]), title="Percentil que dijo el modelo"),
                color=alt.Color("o:N", scale=esc_origen, legend=alt.Legend(orient="top", title=None, columns=2)),
                # Sin tooltip, Vega no detecta el punto bajo el mouse y la ficha no se actualiza;
                # la página desactiva la ventanita de este gráfico (la ficha ya muestra todo).
                tooltip=alt.Tooltip("n:N")))
diag = (alt.Chart(alt.Data(values=[{"x": 0, "y": 0}, {"x": 100, "y": 100}]))
        .mark_line(color="#8a93a0", strokeDash=[5, 4]).encode(x="x:Q", y="y:Q"))
zona = (alt.Chart(alt.Data(values=[{"t": "Arriba de la línea: el modelo lo pone más caro"}]))
        .mark_text(align="left", baseline="top", color="#8a93a0", fontSize=11).encode(x=alt.value(8), y=alt.value(6), text="t:N"))
graf_nube = (diag + nube + zona).properties(width="container", height=420)
linea_tiempo = (base_e.transform_filter(filtro)
                .mark_bar(width=4, opacity=0.9)
                .encode(x=alt.X("yearmonth(fecha):T", title="Lanzamiento de la colección (arrastrá para elegir una época)",
                                axis=alt.Axis(format="%Y", tickCount=12)),
                        y=alt.Y("count():Q", title="Skins"),
                        color=alt.Color("o:N", scale=esc_origen, legend=None))
                .add_params(epoca).properties(width="container", height=110))
graf_explora = alt.vconcat(graf_nube, linea_tiempo, spacing=18).add_params(p_origen, p_rareza, p_arma, p_lado) \
    .resolve_scale(color="shared")

# 11.8 probá el modelo: dónde cae el skin fabricado en la escalera del mercado
p_pred = alt.param(name="pred", value=50)
p_lo = alt.param(name="lo", value=35)
p_hi = alt.param(name="hi", value=65)
fondo = (alt.Chart(skins).transform_calculate(rareza=rareza_nombre)
         .mark_tick(orient="vertical", thickness=1, opacity=0.35)
         .encode(x=alt.X("q:Q", scale=alt.Scale(domain=[0, 100]), title="Posición en el mercado (percentil)"),
                 color=alt.Color("rareza:N", scale=esc_rareza, legend=None)))
uno = alt.Chart(alt.Data(values=[{}]))
banda = uno.transform_calculate(a="lo", b="hi").mark_rect(color="#e8962e", opacity=0.18).encode(x="a:Q", x2="b:Q")
marca = uno.transform_calculate(a="pred").mark_rule(color="#e8962e", strokeWidth=3).encode(x="a:Q")
punta = uno.transform_calculate(a="pred").mark_point(shape="triangle-down", size=260, filled=True, color="#e8962e") \
    .encode(x="a:Q", y=alt.value(-6))
graf_proba = (fondo + banda + marca + punta).add_params(p_pred, p_lo, p_hi).properties(width="container", height=90)

ESPECS = {k: v.to_dict() for k, v in {
    "tira": tira, "datos": graf_datos, "hallazgo": graf_hallazgo, "modelo": graf_modelo, "armory": graf_armory,
    "comparacion": graf_comp, "importancia": graf_imp, "explora": graf_explora, "proba": graf_proba}.items()}

# ================================================================ 12 · textos y página
def n(x, d=0):
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x):
    return f"{n(x)} %"


def con_mayuscula(texto):
    return re.sub(r"^((?:<[^>]+>)*)([a-záéíóúñ])", lambda m: m.group(1) + m.group(2).upper(), str(texto))


fila_tabla = lambda celdas: "<tr>" + "".join(f"<td>{con_mayuscula(c)}</td>" for c in celdas) + "</tr>"
T = {}
T["SNAPSHOT"] = date.fromisoformat(SNAPSHOT).strftime("%d/%m/%Y")
T["FILAS"] = n(len(df))
T["COLECCIONES"] = n(df["collection"].nunique())
T["SKINS_BASE"] = n(df["skin_id"].nunique())
T["ARMAS_N"] = n(len(ARMAS))
T["PRECIO_MIN"] = n(df["sell_price_usd"].min(), 2)
T["PRECIO_MAX"] = n(df["sell_price_usd"].max(), 0)
T["PRECIO_MED"] = n(mediana_precio, 2)
T["VECES_RANGO"] = n(df["sell_price_usd"].max() / df["sell_price_usd"].min(), 0)
T["COBERTURA"] = CALIDAD.get("cobertura", "—").replace(".", ",")
T["MERCADO_ARMAS"] = n(CALIDAD.get("items_de_arma_en_el_mercado", EMBUDO.get("steam_items_de_arma", 0)))
T["DATOS_FRESCOS"] = "sí" if CALIDAD.get("datos_frescos") else "no"
e = EMBUDO
T["EMBUDO"] = "".join(fila_tabla(c) for c in [
    ("Ítems de arma publicados en Steam ese día", "—", n(e.get("steam_items_de_arma", 0)), "Es el universo: la consulta ya pide sólo rifles, pistolas, subfusiles y pesadas"),
    ("Muestra sistemática: páginas repartidas por todo el catálogo", "—", n(e.get("steam_items_leidos", 0)),
     f"{n(e.get('steam_paginas', 0))} páginas de 10 ítems, en proporción a cada categoría"),
    ("Duplicados por la paginación en vivo", "−" + n(e.get("steam_items_leidos", 0) - e.get("steam_sin_duplicados", 0)),
     n(e.get("steam_sin_duplicados", 0)), "El mercado se mueve mientras se pagina: un ítem puede aparecer en dos páginas"),
    ("Sin precio o con precio ≤ 0", "−" + n(e.get("steam_sin_duplicados", 0) - e.get("steam_con_precio", 0)),
     n(e.get("steam_con_precio", 0)), "Un ítem sin precio no sirve para predecir precio"),
    ("Unión con el catálogo de ByMykel por market_hash_name", "−" + n(e.get("steam_con_precio", 0) - e.get("unidos", 0)),
     n(e.get("unidos", 0)), f"El catálogo tiene {n(e.get('catalogo_armas', 0))} ítems de arma; todos los precios encontraron su ficha"),
    ("Validación: clave única, colección en el 100 %, más de 1.000 filas", "0", n(len(df)), "Si algo falla, el pipeline no publica"),
])
T["PEDAZOS"] = "".join(fila_tabla((p["pedazo"], n(p["entrena"]), n(p["valida"]), f"{p['desde']} a {p['hasta']}"))
                       for p in tabla_pedazos)
T["N_TRAIN"], T["N_TEST"] = n(len(train)), n(len(test))
T["COL_TRAIN"], T["COL_TEST"] = n(train["collection"].nunique()), n(test["collection"].nunique())
T["TEST_DESDE"] = f"{test['collection_release'].min():%m/%Y}"

T["H2_D"], T["H2_PARES"], T["H2_MED"] = n(H2["d"], 2), n(H2["pares"]), n(H2["mediana"], 0)
T["H3_RHO"], T["H3_SKINS"], T["H3_BAJA"] = n(H3["rho"], 2), n(H3["skins"]), n(H3["baja"])
T["H4_R"], T["H4_FILAS"] = n(H4["r"], 2), n(H4["filas"])
T["MED_COVERT"], T["MED_CONSUMER"] = n(MEDIANA_RAREZA[5]), n(MEDIANA_RAREZA[0])
T["MED_MILSPEC"], T["MED_CLASSIFIED"] = n(MEDIANA_RAREZA[2]), n(MEDIANA_RAREZA[4])
T["SPEARMAN_RAREZA"] = n(SPEARMAN_RAREZA, 2)
T["MED_MAPA_CONS"], T["MED_CAJA_MIL"] = n(MED_RO[("drop de mapa", 0)]), n(MED_RO[("caja", 2)])
T["MED_CAJA_COV"], T["MED_SOUV_COV"] = n(MED_RO[("caja", 5)]), n(MED_RO[("paquete souvenir", 5)])
T["MED_TODOS_CONS"], T["MED_TODOS_MIL"] = n(MEDIANA_RAREZA[0]), n(MEDIANA_RAREZA[2])
T["H4_SEMAFORO"] = ("rojo en el semáforo (menos de 0,2): no hay efecto que afirmar" if abs(H4["r"]) < 0.2 else
                    "amarillo en el semáforo (entre 0,2 y 0,6): no alcanza para afirmarla" if abs(H4["r"]) < 0.6 else
                    "verde en el semáforo")

T["ELEGIDO"] = ELEGIDO
T["R2_CV"], T["DESVIO_CV"] = n(R2_CV, 2), n(DESVIO_CV, 2)
T["R2_CV3"], T["DESVIO_CV3"] = n(R2_CV, 3), n(DESVIO_CV, 3)
T["R2_CV_PCT"] = pct(100 * R2_CV)
T["R2_PISO"] = n(PISO["r2"], 2)
T["R2_TABLA"] = n(MEJOR_PISO["r2"], 2)
T["R2_TABLA_PCT"] = pct(100 * MEJOR_PISO["r2"])
T["R2_TEST"] = n(R2_TEST, 2)
T["R2_TABLA_TEST"] = n(PRUEBA[1]["prueba"], 2)
T["R2_PISO_TEST"] = n(PRUEBA[0]["prueba"], 2)
T["MAE_CV"], T["MAE_TEST"] = n(MAE_CV, 1), n(MAE_TEST, 1)
T["SESGO_CV"], T["SESGO_TEST"] = n(SESGO_CV, 1), n(SESGO_TEST, 0)
T["SESGO_TEST1"] = n(SESGO_TEST, 1)
T["SESGO_PISO_TEST"] = n(PRUEBA[0]["sesgo_prueba"], 1)
T["COL_SESGO_POS"] = f"{COLECCIONES_SESGO_POS} de {test['collection'].nunique()}"
T["MAS_CARO"] = pct(USD["mas_caro"])
T["USD_MED"], T["USD_MEDIO"] = n(USD["abs_mediano"], 2), n(USD["abs_medio"], 2)
T["USD_VECES"], T["USD_X2"] = n(USD["veces_mediana"], 1), pct(USD["dentro_x2"])
T["USD_REAL"], T["USD_PRED"] = n(USD["real_mediano"], 2), n(USD["pred_mediano"], 2)
T["MAPA_TR"], T["MAPA_TE"] = n(next(c["mediana"] for c in corrimiento if c["origen"] == "drop de mapa" and c["lado"] == "entrenamiento"), 1), \
    n(next(c["mediana"] for c in corrimiento if c["origen"] == "drop de mapa" and c["lado"] == "prueba"), 1)
T["MAPA_SESGO"] = n(next(c["sesgo"] for c in corrimiento if c["origen"] == "drop de mapa" and c["lado"] == "prueba"), 1)
T["MAPA_COL_TR"], T["MAPA_COL_TE"] = str(MAPA["col_tr"]), str(MAPA["col_te"])
T["MAPA_ULTIMA"], T["MAPA_PRIMERA"] = MAPA["ultima_tr"], MAPA["primera_te"]
T["MEDIANA_TEST"] = n(test["percentil"].median(), 0)
T["DUELOS"] = "".join(fila_tabla((d["retador"], ("+" if d["dif"] >= 0 else "−") + n(abs(d["dif"]), 3),
                                  n(d["desvio_dif"], 3), f"{d['gana']} de {len(PEDAZOS)}",
                                  "<b>lo reemplaza</b>" if d["supera"] and d["retador"] == ELEGIDO else "no alcanza"))
                      for d in duelos)
T["COMPARACION"] = "".join(fila_tabla((r["modelo"], n(r["r2"], 3), n(r["desvio"], 3), n(r["mae"], 1), "+" + n(r["sesgo"], 1)))
                           for r in sorted(pisos + comparacion, key=lambda r: -r["r2"]))
T["MEJORES"] = ", ".join(f"<code>{k}={v}</code>" for k, v in MEJORES.items())
T["N_GRILLA"] = str(N_GRILLA)
T["AJUSTE_DIF"], T["AJUSTE_DESV"] = n(AJUSTE["dif"], 3), n(AJUSTE["desvio_dif"], 3)
T["AJUSTE_LECTURA"] = ("sube por más que el desvío de la diferencia, así que vale la pena"
                       if AJUSTE["dif"] > AJUSTE["desvio_dif"] else "queda dentro del ruido")
T["TRAMOS"] = "".join(fila_tabla((str(t.tramo), n(t.filas), n(t.mae, 1), ("+" if t.sesgo >= 0 else "−") + n(abs(t.sesgo), 1),
                                  "USD " + n(t.real, 2), "USD " + n(t.pred, 2), "USD " + n(t.usd, 2)))
                      for t in por_tramo.itertuples())
T["PEORES"] = "".join(fila_tabla((r.market_hash_name, r.collection.replace("The ", "").replace(" Collection", ""), r.origen,
                                  n(r.percentil, 0), n(r.predicho, 0), "USD " + n(r.sell_price_usd, 2),
                                  "USD " + n(r.precio_predicho, 2)))
                      for r in peores.itertuples())
T["IMP_1"], T["IMP_1_V"] = IMPORTANCIA[0]["columna"], n(IMPORTANCIA[0]["baja"], 2)
T["IMP_2"], T["IMP_2_V"] = IMPORTANCIA[1]["columna"], n(IMPORTANCIA[1]["baja"], 2)
T["R2_OOF"] = n(R2_OOF, 2)
T["VERIFICADO"] = VERIFICADO
T["SKLEARN"] = sklearn.__version__
T["N_GRILLA_APP"] = n(len(grilla))

JS = {"SKINS": SKINS, "ESPECS": ESPECS, "GRILLA": GRILLA, "VISTOS": VISTOS, "ARMAS": ARMAS, "ORIGENES": ORIGENES,
      "CUANTILES": CUANTILES_PRECIO, "MAE_CV": round(MAE_CV, 2), "SESGO_MAPA": round(float(T["MAPA_SESGO"].replace(",", ".")), 1),
      "COL_RAREZA": COL_RAREZA, "COL_ORIGEN": COL_ORIGEN}

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
                  "r2_oof": R2_OOF, "mediana_rareza": MEDIANA_RAREZA}, default=str, indent=1))
