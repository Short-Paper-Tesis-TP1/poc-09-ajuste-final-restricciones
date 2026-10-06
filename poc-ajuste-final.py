"""
PoC de ajuste final del diccionario
===================================
Objetivo único: elegir la configuración final de variables y restricciones del modelo v1.

Preguntas que responde:
  A. ¿El costo de AUC de las restricciones monotónicas viene de ratio_monto_ingreso?
     La Res. SBS 6941-2008 (art. 5) respalda como indicadores de riesgo: cuota/ingreso, deuda/ingreso,
     número de entidades y comportamiento de pago. NO menciona monto/ingreso.
     Regla: restringir SOLO las 4 variables con respaldo SBS.
  B. ¿Cuánto suman las candidatas combinadas?
     Incluye "plazo_meses" = monto / cuota, que se DERIVA de las preguntas 2 y 3 (no agrega preguntas).

Regla de decisión (fijada ANTES de correr):
  - Una combinación se prefiere a otra solo si mejora el AUC en >= 0.005 en todas Y en emprendedoras.
  - A igualdad (diferencia < 0.005), se prefiere la de MENOS preguntas.
  - Toda variable nueva necesita además sustento Q1-Q2 para entrar al diccionario.

Uso: python poc_ajuste_final.py "C:\\ruta\\home-credit-default-risk"
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RUTA = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\alexa\Escritorio\UPC\TP1\home-credit-default-risk"
SEMILLA = 42
EDUCACION = {"Lower secondary": 0, "Secondary / secondary special": 1, "Incomplete higher": 2,
             "Higher education": 3, "Academic degree": 3}


def csv(nombre):
    return os.path.join(RUTA, f"dataset-{nombre}.csv")


t0 = time.time()
app = pd.read_csv(csv("application_train"), usecols=[
    "SK_ID_CURR", "TARGET", "CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT", "DAYS_BIRTH",
    "NAME_EDUCATION_TYPE", "CNT_CHILDREN", "NAME_INCOME_TYPE", "ORGANIZATION_TYPE", "DAYS_EMPLOYED"])
w = app[app.CODE_GENDER == "F"].reset_index(drop=True)

bur = pd.read_csv(csv("bureau"), usecols=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE",
                                          "AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT"])
en_buro = w.SK_ID_CURR.isin(bur.SK_ID_CURR)
act = (bur[bur.CREDIT_ACTIVE == "Active"].assign(d=lambda x: x.AMT_CREDIT_SUM_DEBT.fillna(0))
       .groupby("SK_ID_CURR").agg(n=("SK_ID_BUREAU", "count"), deuda=("d", "sum")))
hist = bur.groupby("SK_ID_CURR").agg(dias_primero=("DAYS_CREDIT", "min"))
w = w.merge(act, left_on="SK_ID_CURR", right_index=True, how="left") \
     .merge(hist, left_on="SK_ID_CURR", right_index=True, how="left")

bb = pd.read_csv(csv("bureau_balance"), usecols=["SK_ID_BUREAU", "STATUS"],
                 dtype={"SK_ID_BUREAU": "int64", "STATUS": "category"})
ids_bb = bb.SK_ID_BUREAU.unique()
ids_atr = bb.loc[bb.STATUS.isin(["1", "2", "3", "4", "5"]), "SK_ID_BUREAU"].unique()
del bb
c_bb = bur.loc[bur.SK_ID_BUREAU.isin(ids_bb), "SK_ID_CURR"].unique()
c_atr = bur.loc[bur.SK_ID_BUREAU.isin(ids_atr), "SK_ID_CURR"].unique()

prev = pd.read_csv(csv("previous_application"), usecols=["SK_ID_CURR", "NAME_CONTRACT_STATUS", "DAYS_DECISION"])
rechazos = prev[prev.NAME_CONTRACT_STATUS == "Refused"].groupby("SK_ID_CURR").size()
# Solicitudes de crédito en los últimos 12 meses (cualquier resultado): equivalente a "consultas recientes"
solicitudes_12m = prev[prev.DAYS_DECISION >= -365].groupby("SK_ID_CURR").size()

# --- Variables (orden del diccionario) ---
V = pd.DataFrame(index=w.index)
V["ratio_cuota_ingreso"] = w.AMT_ANNUITY / w.AMT_INCOME_TOTAL
V["ratio_deuda_ingreso"] = (w.AMT_CREDIT + np.where(en_buro, w.deuda.fillna(0), np.nan)) / w.AMT_INCOME_TOTAL
V["ratio_monto_ingreso"] = w.AMT_CREDIT / w.AMT_INCOME_TOTAL
V["n_otros_creditos"] = np.where(en_buro, w.n.fillna(0), np.nan)
V["atraso_historial"] = np.where(w.SK_ID_CURR.isin(c_atr), 1.0, np.where(w.SK_ID_CURR.isin(c_bb), 0.0, np.nan))
V["edad"] = -w.DAYS_BIRTH / 365.25
V["nivel_educativo"] = w.NAME_EDUCATION_TYPE.map(EDUCACION)
V["n_hijos"] = w.CNT_CHILDREN
V["anios_trabajo_negocio"] = -w.DAYS_EMPLOYED.where(w.DAYS_EMPLOYED != 365243) / 365.25
V["anios_historial_credito"] = np.where(en_buro, -w.dias_primero / 365.25, np.nan)
V["n_rechazos"] = w.SK_ID_CURR.map(rechazos).fillna(0)
V["plazo_meses"] = w.AMT_CREDIT / w.AMT_ANNUITY           # derivada de preguntas 2 y 3
V["n_solicitudes_12m"] = w.SK_ID_CURR.map(solicitudes_12m).fillna(0)  # ¿Cuántos créditos pediste en el último año?

BASE = ["ratio_cuota_ingreso", "ratio_deuda_ingreso", "ratio_monto_ingreso", "n_otros_creditos",
        "atraso_historial", "edad", "nivel_educativo", "n_hijos"]
SBS = {"ratio_cuota_ingreso", "ratio_deuda_ingreso", "n_otros_creditos", "atraso_historial"}
y = w.TARGET.values
emp = ((w.NAME_INCOME_TYPE == "Commercial associate") | (w.ORGANIZATION_TYPE == "Self-employed")).values


def auc_cv(cols, restringidas):
    mono = [1 if c in restringidas else 0 for c in cols]
    X = V[cols]
    todas, empr = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEMILLA).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                               monotone_constraints=mono, random_state=SEMILLA, verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        todas.append(roc_auc_score(y[te], p))
        empr.append(roc_auc_score(y[te][emp[te]], p[emp[te]]))
    return {"variables": cols, "restringidas": sorted(set(cols) & set(restringidas)),
            "auc_todas": float(np.mean(todas)), "auc_emprendedoras": float(np.mean(empr))}


configs = {
    "A1 BASE, 5 restricciones (actual)": (BASE, SBS | {"ratio_monto_ingreso"}),
    "A2 BASE, solo 4 restricciones SBS": (BASE, SBS),
    "A3 BASE, sin restricciones": (BASE, set()),
    "B1 SBS + años trabajo/negocio": (BASE + ["anios_trabajo_negocio"], SBS),
    "B2 SBS + años historial crediticio": (BASE + ["anios_historial_credito"], SBS),
    "B3 SBS + rechazos": (BASE + ["n_rechazos"], SBS),
    "B4 SBS + plazo (sin pregunta nueva)": (BASE + ["plazo_meses"], SBS),
    "B5 SBS + solicitudes último año": (BASE + ["n_solicitudes_12m"], SBS),
    "C1 SBS + trabajo + historial": (BASE + ["anios_trabajo_negocio", "anios_historial_credito"], SBS),
    "C2 SBS + trabajo + historial + plazo": (BASE + ["anios_trabajo_negocio", "anios_historial_credito",
                                                     "plazo_meses"], SBS),
    "C3 SBS + trabajo + historial + rechazos": (BASE + ["anios_trabajo_negocio", "anios_historial_credito",
                                                        "n_rechazos"], SBS),
    "C4 SBS + las cuatro": (BASE + ["anios_trabajo_negocio", "anios_historial_credito", "n_rechazos",
                                    "plazo_meses"], SBS),
    "C5 SBS + trabajo + plazo + solicitudes": (BASE + ["anios_trabajo_negocio", "plazo_meses",
                                                       "n_solicitudes_12m"], SBS),
    "C6 SBS + trabajo + plazo": (BASE + ["anios_trabajo_negocio", "plazo_meses"], SBS),
}
res = {k: auc_cv(*v) for k, v in configs.items()}
ref = res["A2 BASE, solo 4 restricciones SBS"]
for r in res.values():
    r["vs_A2_todas"] = r["auc_todas"] - ref["auc_todas"]
    r["vs_A2_emprendedoras"] = r["auc_emprendedoras"] - ref["auc_emprendedoras"]
    r["preguntas_nuevas"] = len([c for c in r["variables"] if c not in BASE and c != "plazo_meses"])

print(f"\nMujeres: {len(V):,} | emprendedoras: {emp.sum():,} | {time.time()-t0:.0f} s")
print(f"\n{'Configuración':42s} {'AUC todas':>9s} {'vs A2':>7s} {'AUC empr':>9s} {'vs A2':>7s} {'preg. nuevas':>12s}")
for k, r in res.items():
    print(f"{k:42s} {r['auc_todas']:9.4f} {r['vs_A2_todas']:+7.4f} {r['auc_emprendedoras']:9.4f} "
          f"{r['vs_A2_emprendedoras']:+7.4f} {r['preguntas_nuevas']:12d}")
with open("resultado_poc_ajuste_final.json", "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=False)
print("\nGuardado: resultado_poc_ajuste_final.json")