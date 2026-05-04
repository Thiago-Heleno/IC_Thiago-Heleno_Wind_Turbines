"""
Gera figuras do relatorio IC a partir de:
  - resultados/*/metricas.json
  - resultados/metricas_consolidadas.csv
  - CARE_To_Compare/Wind Farm C/datasets (sample EDA)
  - analise_comparativa.md (NB1 hardcoded, ainda pendente em json)
Saida: figs/geradas/*.png  (300 dpi)
"""
from __future__ import annotations
import json, os, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resultados"
OUT = ROOT / "figs" / "geradas"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "CARE_To_Compare" / "Wind Farm C"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. Resultados consolidados (NB1 hardcoded de analise_comparativa.md)
# ─────────────────────────────────────────────────────────────────────────────
NB1 = {
    "CNN":      {"precision": 0.1962, "recall": 0.0318, "f1": 0.0547, "auc_roc": 0.55, "accuracy": 0.9815},
    "LSTM":     {"precision": 0.0,    "recall": 0.0,    "f1": 0.0,    "auc_roc": 0.40, "accuracy": 0.9831},
    "CNN-LSTM": {"precision": 0.0328, "recall": 0.0305, "f1": 0.0316, "auc_roc": 0.66, "accuracy": 0.9685},
}
NB1_TEST = dict(total=541109, pos=9122, neg=531987)
# NB2/3/5 — split temporal CARE
NB23_TEST = dict(total=478076, pos=6366, neg=471710)

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

m2 = load_json(RES / "02_cnn_bilstm_autoencoder" / "metricas.json")
m3 = load_json(RES / "03_keras_mlp_autoencoder" / "metricas.json")
m4 = load_json(RES / "04_classifier_induced_failure" / "metricas.json")
m5 = load_json(RES / "05_autoencoder_induced_failure" / "metricas.json")


# ─────────────────────────────────────────────────────────────────────────────
# Fig A — Comparacao F1 entre todos modelos
# ─────────────────────────────────────────────────────────────────────────────
def fig_f1_comparativo():
    rows = [
        ("NB1 CNN",              NB1["CNN"]["f1"],      "Sup."),
        ("NB1 LSTM",             NB1["LSTM"]["f1"],     "Sup."),
        ("NB1 CNN-LSTM",         NB1["CNN-LSTM"]["f1"], "Sup."),
        ("NB2 P95",              m2["metricas_teste"]["v3_P95_baseline"]["f1"], "Semi-sup."),
        ("NB2 P99",              m2["metricas_teste"]["v3_P99_baseline"]["f1"], "Semi-sup."),
        ("NB2 v4 ScorePond",     m2["metricas_teste"]["v4_ScorePond_BF1"]["f1"], "Semi-sup."),
        ("NB4 Induzido",         m4["metricas_teste"]["induced_model"]["f1"],  "Sup. (sint.)"),
        ("NB4 Baseline",         m4["metricas_teste"]["baseline"]["f1"],       "Sup. (sint.)"),
        ("NB5 Standard P95",     m5["metricas_teste"]["Standard (P95)"]["f1"],         "Semi-sup. (sint.)"),
        ("NB5 Induced P95",      m5["metricas_teste"]["Induced (P95 induced)"]["f1"], "Semi-sup. (sint.)"),
        ("NB5 Adaptive",         m5["metricas_teste"]["Adaptive (P50_ind+gamma)"]["f1"], "Semi-sup. (sint.)"),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    paradigmas = [r[2] for r in rows]
    cmap = {"Sup.":"#377eb8","Semi-sup.":"#4daf4a","Sup. (sint.)":"#984ea3","Semi-sup. (sint.)":"#ff7f00"}
    cores = [cmap[p] for p in paradigmas]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=cores, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("F1-Score (amostra)")
    ax.set_title("Comparativo F1-Score por Configuracao (conjunto de teste)")
    ax.set_ylim(0, max(vals)*1.15 + 0.05)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    handles = [Rectangle((0,0),1,1,color=c) for c in cmap.values()]
    ax.legend(handles, list(cmap.keys()), loc="upper left", fontsize=9)
    fig.savefig(OUT/"f1_comparativo.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig B — Comparacao AUC-ROC
# ─────────────────────────────────────────────────────────────────────────────
def fig_auc_comparativo():
    rows = [
        ("NB1 CNN", NB1["CNN"]["auc_roc"]),
        ("NB1 LSTM", NB1["LSTM"]["auc_roc"]),
        ("NB1 CNN-LSTM", NB1["CNN-LSTM"]["auc_roc"]),
        ("NB2 v3 baseline", m2["metricas_teste"]["v3_P95_baseline"]["auc_roc"]),
        ("NB2 v4 ScorePond", m2["metricas_teste"]["v4_ScorePond_BF1"]["auc_roc"]),
        ("NB3 Keras AE",  0.8717),  # de analise_comparativa
    ]
    labels=[r[0] for r in rows]; vals=[r[1] for r in rows]
    fig,ax=plt.subplots(figsize=(8.5,4.5))
    x=np.arange(len(labels))
    cores=["#377eb8" if "NB1" in l else "#4daf4a" for l in labels]
    bars=ax.bar(x,vals,color=cores,edgecolor="black",linewidth=0.6)
    ax.axhline(0.5,color="red",ls="--",lw=1,label="Aleatorio (AUC=0,5)")
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=25,ha="right")
    ax.set_ylabel("AUC-ROC"); ax.set_ylim(0,1.0)
    ax.set_title("Comparativo AUC-ROC por modelo")
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,v+0.015,f"{v:.3f}",ha="center",fontsize=9)
    ax.legend(loc="lower right")
    fig.savefig(OUT/"auc_comparativo.png"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig C — Curvas ROC sinteticas (binormal a partir de AUC)
# ─────────────────────────────────────────────────────────────────────────────
def binormal_roc(auc, n=400):
    """Aproxima ROC binormal: assumindo distribuicoes normais com mesma variancia,
    d = sqrt(2)*Phi^-1(AUC). Curva: TPR = Phi(Phi^-1(FPR) + d)."""
    if auc <= 0.5:
        # gerar curva diagonal/refletida
        fpr = np.linspace(0,1,n); tpr = fpr.copy()
        return fpr, tpr
    d = math.sqrt(2)*stats.norm.ppf(auc)
    fpr = np.linspace(1e-4,1-1e-4,n)
    tpr = stats.norm.cdf(stats.norm.ppf(fpr)+d)
    return np.concatenate([[0],fpr,[1]]), np.concatenate([[0],tpr,[1]])

def fig_roc_curvas():
    modelos = [
        ("NB1 CNN",      NB1["CNN"]["auc_roc"]),
        ("NB1 LSTM",     NB1["LSTM"]["auc_roc"]),
        ("NB1 CNN-LSTM", NB1["CNN-LSTM"]["auc_roc"]),
        ("NB2 v4 (CNN-BiLSTM-Att AE)", m2["metricas_teste"]["v4_ScorePond_BF1"]["auc_roc"]),
        ("NB3 Keras MLP AE",           0.8717),
    ]
    fig,ax=plt.subplots(figsize=(7.5,6))
    cores = plt.cm.tab10(np.linspace(0,1,len(modelos)))
    for (nome,auc),c in zip(modelos,cores):
        fpr,tpr = binormal_roc(auc)
        ax.plot(fpr,tpr,label=f"{nome} (AUC={auc:.3f})",color=c,lw=1.6)
    ax.plot([0,1],[0,1],"k--",lw=1,label="Aleatorio")
    ax.set_xlabel("Taxa de Falso Positivo (FPR)")
    ax.set_ylabel("Taxa de Verdadeiro Positivo (TPR)")
    ax.set_title("Curvas ROC (aproximadas, modelo binormal a partir de AUC)")
    ax.legend(loc="lower right",fontsize=9)
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    fig.text(0.02,0.02,"Curvas reconstruidas a partir do AUC reportado (scores brutos nao foram salvos).",
             fontsize=7,style="italic",color="dimgray")
    fig.savefig(OUT/"roc_curvas_aproximadas.png"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig D — Matrizes de confusao (reconstruidas)
# ─────────────────────────────────────────────────────────────────────────────
def cm_from_pr(precision, recall, real_pos, real_neg):
    tp = recall * real_pos
    fn = real_pos - tp
    if precision > 1e-6:
        fp = tp/precision - tp
    else:
        fp = 0.0  # modelo nao previu positivo
    tn = real_neg - fp
    return np.array([[tn,fp],[fn,tp]]).round().astype(int)

def cm_from_pra(precision, recall, accuracy, total):
    """Deriva CM quando a base de teste e desconhecida (NB4/NB5: subset balanceado).
    Resolve prevalencia de positivos a partir de p, r, acc."""
    if precision <= 1e-6 or recall <= 1e-6:
        # modelo colapsou — usa balanced 50/50 como fallback
        pos_prev = 0.5
    else:
        denom = (1-recall) + recall*(1-precision)/precision
        pos_prev = (1-accuracy)/denom if denom>0 else 0.5
        pos_prev = max(0.01, min(0.99, pos_prev))
    real_pos = total*pos_prev
    real_neg = total*(1-pos_prev)
    return cm_from_pr(precision, recall, real_pos, real_neg)

def plot_cm(ax, cm, titulo):
    norm = cm/cm.sum()
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred Normal","Pred Anomalia"])
    ax.set_yticklabels(["Real Normal","Real Anomalia"])
    for i in range(2):
        for j in range(2):
            txt = f"{cm[i,j]:,}\n({norm[i,j]*100:.2f}%)"
            ax.text(j,i,txt,ha="center",va="center",
                    color="white" if norm[i,j]>0.5 else "black",fontsize=10)
    ax.set_title(titulo,fontsize=11)

def fig_matrizes_confusao():
    # NB1/NB2: usa contagens reais do split CARE
    # NB4/NB5: subset balanceado (acc reportada) — deriva via cm_from_pra
    nb4i = m4["metricas_teste"]["induced_model"]
    nb4b = m4["metricas_teste"]["baseline"]
    nb5s = m5["metricas_teste"]["Standard (P95)"]
    nb5i = m5["metricas_teste"]["Induced (P95 induced)"]
    casos_pr = [
        ("NB1 CNN",      NB1["CNN"]["precision"], NB1["CNN"]["recall"], NB1_TEST),
        ("NB1 LSTM",     NB1["LSTM"]["precision"],NB1["LSTM"]["recall"],NB1_TEST),
        ("NB1 CNN-LSTM", NB1["CNN-LSTM"]["precision"],NB1["CNN-LSTM"]["recall"],NB1_TEST),
        ("NB2 v4 ScorePond", m2["metricas_teste"]["v4_ScorePond_BF1"]["precision"],
                              m2["metricas_teste"]["v4_ScorePond_BF1"]["recall"], NB23_TEST),
    ]
    casos_pra = [
        ("NB4 Induzido (3-cls, sub-balanc.)", nb4i["precision"], nb4i["recall"], nb4i["accuracy"], 100000),
        ("NB5 Induced P95 (sub-balanc.)",     nb5i["precision"], nb5i["recall"], nb5i["accuracy"], 100000),
    ]
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    axs = axs.ravel()
    i=0
    for nome,p,r,t in casos_pr:
        cm = cm_from_pr(p,r,t["pos"],t["neg"])
        plot_cm(axs[i],cm,nome); i+=1
    for nome,p,r,a,total in casos_pra:
        cm = cm_from_pra(p,r,a,total)
        plot_cm(axs[i],cm,nome); i+=1
    fig.suptitle("Matrizes de Confusao reconstruidas (NB1/NB2: split CARE; NB4/NB5: subset balanceado, normalizado para 100k)",y=1.00)
    fig.savefig(OUT/"matrizes_confusao.png"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig E — CARE Score comparativo (NB3, NB4, NB5)
# ─────────────────────────────────────────────────────────────────────────────
def fig_care_comparativo():
    rows=[
        ("NB3 Keras AE",      m3["metricas_CARE"]["CARE"], m3["metricas_CARE"]["EF1_2"], m3["metricas_CARE"]["Acc"], m3["metricas_CARE"]["WS"]),
        ("NB4 Induzido",      m4["metricas_CARE"]["induced"]["CARE"], m4["metricas_CARE"]["induced"]["EF1_2"], m4["metricas_CARE"]["induced"]["Acc"], m4["metricas_CARE"]["induced"]["WS"]),
        ("NB4 Baseline",      m4["metricas_CARE"]["baseline"]["CARE"], m4["metricas_CARE"]["baseline"]["EF1_2"], m4["metricas_CARE"]["baseline"]["Acc"], m4["metricas_CARE"]["baseline"]["WS"]),
        ("NB5 Standard P95",  m5["metricas_CARE"]["Standard (P95)"]["CARE"], m5["metricas_CARE"]["Standard (P95)"]["EF1_2"], m5["metricas_CARE"]["Standard (P95)"]["Acc"], m5["metricas_CARE"]["Standard (P95)"]["WS"]),
        ("NB5 Induced P95",   m5["metricas_CARE"]["Induced (P95 induced)"]["CARE"], m5["metricas_CARE"]["Induced (P95 induced)"]["EF1_2"], m5["metricas_CARE"]["Induced (P95 induced)"]["Acc"], m5["metricas_CARE"]["Induced (P95 induced)"]["WS"]),
        ("NB5 Adaptive",      m5["metricas_CARE"]["Adaptive (P50_ind+gamma)"]["CARE"], m5["metricas_CARE"]["Adaptive (P50_ind+gamma)"]["EF1_2"], m5["metricas_CARE"]["Adaptive (P50_ind+gamma)"]["Acc"], m5["metricas_CARE"]["Adaptive (P50_ind+gamma)"]["WS"]),
    ]
    labels=[r[0] for r in rows]
    care=[r[1] for r in rows]; ef1=[r[2] for r in rows]; acc=[r[3] for r in rows]; ws=[r[4] for r in rows]
    fig,axs=plt.subplots(1,2,figsize=(13,4.6))
    x=np.arange(len(labels))
    axs[0].bar(x,care,color="#1f77b4",edgecolor="black")
    axs[0].set_xticks(x); axs[0].set_xticklabels(labels,rotation=30,ha="right")
    axs[0].set_ylabel("CARE Score"); axs[0].set_title("CARE Score (composto, por evento)")
    for i,v in enumerate(care):
        axs[0].text(i,v+0.005,f"{v:.3f}" if v>1e-3 else f"{v:.1e}",ha="center",fontsize=8)

    w=0.22
    axs[1].bar(x-1.5*w,ef1,w,label="EF1_2")
    axs[1].bar(x-0.5*w,acc,w,label="Acc evento")
    axs[1].bar(x+0.5*w,ws,w,label="WS")
    axs[1].bar(x+1.5*w,care,w,label="CARE")
    axs[1].set_xticks(x); axs[1].set_xticklabels(labels,rotation=30,ha="right")
    axs[1].set_title("Sub-componentes da metrica CARE")
    axs[1].legend(fontsize=8); axs[1].set_ylim(0,1.05)
    fig.savefig(OUT/"care_comparativo.png"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig F — Trade-off precision vs recall NB5 (3 thresholds)
# ─────────────────────────────────────────────────────────────────────────────
def fig_tradeoff_nb5():
    rows=[
        ("Standard (P95)",         m5["thresholds"]["Standard (P95)"],
         m5["metricas_teste"]["Standard (P95)"]["precision"],
         m5["metricas_teste"]["Standard (P95)"]["recall"]),
        ("Induced (P95 induced)", m5["thresholds"]["Induced (P95 induced)"],
         m5["metricas_teste"]["Induced (P95 induced)"]["precision"],
         m5["metricas_teste"]["Induced (P95 induced)"]["recall"]),
        ("Adaptive (P50_ind+γ)",   m5["thresholds"]["Adaptive (P50_ind+gamma)"],
         m5["metricas_teste"]["Adaptive (P50_ind+gamma)"]["precision"],
         m5["metricas_teste"]["Adaptive (P50_ind+gamma)"]["recall"]),
    ]
    fig,ax=plt.subplots(figsize=(7.5,5))
    for nome,thr,p,r in rows:
        ax.scatter(r,p,s=130,label=f"{nome}\n(thr={thr:.3f})")
        ax.annotate(nome.split()[0], (r,p), xytext=(8,8), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("NB5 — Trade-off Precision/Recall por threshold (amostra)")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(loc="lower left",fontsize=8)
    fig.savefig(OUT/"nb5_tradeoff_precision_recall.png"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig G — Comparativo NB3 vs NB5 (pontos opostos curva ROC)
# ─────────────────────────────────────────────────────────────────────────────
def fig_nb3_vs_nb5():
    # NB3 amostra: precision 0.0232, recall 0.8826 (analise_comparativa)
    nb3 = (0.8826, 0.0232, "NB3 Keras AE")
    nb5 = (m5["metricas_teste"]["Induced (P95 induced)"]["recall"],
           m5["metricas_teste"]["Induced (P95 induced)"]["precision"], "NB5 Induced")
    fig,ax=plt.subplots(figsize=(7,5))
    for r,p,nome in [nb3,nb5]:
        ax.scatter(r,p,s=180,label=nome)
        ax.annotate(nome,(r,p),xytext=(6,8),textcoords="offset points",fontsize=10)
    # baseline iso-F1 curves
    for f1 in [0.05,0.1,0.2,0.3,0.5]:
        rs=np.linspace(0.01,0.99,200)
        ps=(f1*rs)/(2*rs-f1)
        ok = (ps>0)&(ps<=1)
        ax.plot(rs[ok],ps[ok],"--",color="gray",alpha=0.4)
        # label
        idx = np.argmax(ok & (rs>0.4))
        if ok[idx]:
            ax.text(rs[idx],ps[idx],f"F1={f1}",color="gray",fontsize=7)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("NB3 vs NB5 — pontos opostos do trade-off (amostra)")
    ax.legend()
    fig.savefig(OUT/"nb3_vs_nb5.png"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# EDA — carrega amostra de datasets
# ─────────────────────────────────────────────────────────────────────────────
def carregar_amostra(n_files=8, n_rows_each=5000):
    files = sorted(glob.glob(str(DATA/"datasets"/"*.csv")))[:n_files]
    parts=[]
    for f in files:
        try:
            df = pd.read_csv(f, sep=";", nrows=n_rows_each, low_memory=False)
            df["__source__"] = os.path.basename(f)
            parts.append(df)
        except Exception as e:
            print("warn:", f, e)
    return pd.concat(parts, ignore_index=True), files

def fig_eda_classes():
    files = sorted(glob.glob(str(DATA/"datasets"/"*.csv")))
    # contagem rapida usando apenas col status_type_id
    counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    for f in files:
        try:
            s = pd.read_csv(f, sep=";", usecols=["status_type_id"], low_memory=False)["status_type_id"]
            for k,v in s.value_counts().items():
                counts[int(k)] = counts.get(int(k),0)+int(v)
        except Exception as e:
            print("warn",f,e)
    # CARE convention: status 0 (Normal Operation) E status 2 (Idling) sao normais.
    # Demais (1=Derated, 3=Service, 4=Downtime, 5=Other) sao nao-normais.
    # OBS: rotulo de anomalia REAL e' por evento (event_label), nao status_type_id.
    normal = counts.get(0,0) + counts.get(2,0)
    anom = sum(v for k,v in counts.items() if k not in (0,2))
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    bars=ax.bar(["Normal (status 0+2)","Nao-normal (status 1,3,4,5)"],
                [normal,anom],
                color=["#4daf4a","#e41a1c"], edgecolor="black")
    total=normal+anom
    for b,v in zip(bars,[normal,anom]):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:,}\n({v/total*100:.2f}%)",
                ha="center",va="bottom",fontsize=10)
    ax.set_ylabel("Numero de amostras (10 min)")
    ax.set_title(f"Desbalanceamento de classes — Wind Farm C ({total:,} amostras totais)")
    ax.set_ylim(0, max(normal,anom)*1.15)
    fig.savefig(OUT/"eda_desbalanceamento.png"); plt.close(fig)
    return counts

def fig_eda_potencia_serie(df_amostra):
    # plota power_2_avg de 1 dataset
    one = df_amostra[df_amostra["__source__"]==df_amostra["__source__"].iloc[0]].copy()
    one["time_stamp"] = pd.to_datetime(one["time_stamp"], errors="coerce")
    one = one.dropna(subset=["time_stamp"]).head(2000)
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(one["time_stamp"], one["power_2_avg"], color="#1f77b4", lw=0.7)
    ax.set_xlabel("Tempo"); ax.set_ylabel("Potencia (power_2_avg, normalizado)")
    ax.set_title(f"Serie temporal de potencia gerada — dataset {one['__source__'].iloc[0]} (primeiras 2000 amostras)")
    fig.autofmt_xdate()
    fig.savefig(OUT/"eda_serie_potencia.png"); plt.close(fig)

def fig_eda_hist_vento(df_amostra):
    fig,ax=plt.subplots(figsize=(7,4.5))
    v = df_amostra["wind_speed_236_avg"].dropna()
    ax.hist(v, bins=60, color="#377eb8", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Velocidade do vento (wind_speed_236_avg)")
    ax.set_ylabel("Frequencia")
    ax.set_title(f"Distribuicao de velocidade do vento (amostra de {len(v):,} registros)")
    fig.savefig(OUT/"eda_hist_vento.png"); plt.close(fig)

def fig_eda_correlacao(df_amostra):
    cols = [c for c in df_amostra.columns
            if c.endswith("_avg") and any(k in c for k in
                ["power_","wind_speed_","sensor_0_","sensor_1_","sensor_3_",
                 "sensor_7_","sensor_14_","sensor_15_","sensor_18_","sensor_144_",
                 "sensor_145_","sensor_236","sensor_5_","sensor_6_","sensor_17_"])]
    cols = cols[:18]
    sub = df_amostra[cols].apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)
    corr = sub.corr().values
    fig,ax=plt.subplots(figsize=(8,7))
    im=ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sub.columns))); ax.set_yticks(range(len(sub.columns)))
    ax.set_xticklabels(sub.columns, rotation=80, fontsize=7)
    ax.set_yticklabels(sub.columns, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Matriz de correlacao — subset de {len(sub.columns)} variaveis SCADA (avg)")
    fig.savefig(OUT/"eda_matriz_correlacao.png"); plt.close(fig)

def fig_eda_missing(df_amostra):
    """No SCADA do CARE_To_Compare, sensor inativo aparece como valor 0 fixo
    (nao NaN). Mapa abaixo: fracao de valores == 0 por feature/dataset (proxy
    de 'missing' / sensor offline) usando MAIS arquivos e sample maior."""
    files = sorted(glob.glob(str(DATA/"datasets"/"*.csv")))[:18]
    cols_avg = None
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=";", nrows=20000, low_memory=False)
        except Exception as e:
            print("warn",f,e); continue
        if cols_avg is None:
            cols_avg = [c for c in df.columns if c.endswith("_avg")][:80]
        sub = df[cols_avg].apply(pd.to_numeric, errors="coerce")
        frac_zero = (sub==0).mean().values
        frac_nan  = sub.isna().mean().values
        rows.append((os.path.basename(f).replace(".csv",""),
                     np.maximum(frac_zero, frac_nan)))
    sources = [r[0] for r in rows]
    mat = np.array([r[1] for r in rows])
    fig,ax=plt.subplots(figsize=(12,5.5))
    im=ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1.0)
    ax.set_yticks(range(len(sources))); ax.set_yticklabels(sources, fontsize=8)
    ax.set_xticks([]); ax.set_xlabel(f"{len(cols_avg)} features SCADA (cols *_avg)")
    ax.set_ylabel("Dataset")
    plt.colorbar(im,ax=ax,label="Fracao zero/NaN (sensor inativo)")
    ax.set_title("Mapa de calor de sensores inativos (0/NaN) por dataset x feature")
    fig.savefig(OUT/"eda_missing.png"); plt.close(fig)

def fig_eda_boxplot_outliers(df_amostra):
    cols = ["power_2_avg","wind_speed_236_avg","sensor_0_avg","sensor_3_avg","sensor_18_avg","sensor_144_avg"]
    data = [pd.to_numeric(df_amostra[c],errors="coerce").dropna() for c in cols]
    # clipping Q0.1/Q99.9 (mesma estrategia DataClipper)
    data_clip=[]
    for s in data:
        lo,hi = s.quantile([0.001,0.999])
        data_clip.append(s.clip(lo,hi))
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=False)
    axs[0].boxplot(data, labels=cols, showfliers=True)
    axs[0].set_title("Antes do clipping (com outliers)")
    axs[0].tick_params(axis="x",rotation=30)
    axs[1].boxplot(data_clip, labels=cols, showfliers=True)
    axs[1].set_title("Depois do clipping Q0.1%–Q99.9%")
    axs[1].tick_params(axis="x",rotation=30)
    fig.suptitle("Tratamento de outliers — boxplot antes/depois")
    fig.savefig(OUT/"eda_boxplot_outliers.png"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(">> figuras quantitativas")
    fig_f1_comparativo()
    fig_auc_comparativo()
    fig_roc_curvas()
    fig_matrizes_confusao()
    fig_care_comparativo()
    fig_tradeoff_nb5()
    fig_nb3_vs_nb5()

    print(">> EDA: contagem classes (lendo todos os 58 datasets, so col status)")
    fig_eda_classes()

    print(">> EDA: amostra (8 datasets x 5000 linhas)")
    df_amostra,_ = carregar_amostra(n_files=8, n_rows_each=5000)
    print(f"   shape amostra: {df_amostra.shape}")
    fig_eda_potencia_serie(df_amostra)
    fig_eda_hist_vento(df_amostra)
    fig_eda_correlacao(df_amostra)
    fig_eda_missing(df_amostra)
    fig_eda_boxplot_outliers(df_amostra)

    figs = sorted(OUT.glob("*.png"))
    print(f"\n>> {len(figs)} figuras geradas em {OUT}")
    for f in figs:
        print("  -",f.name)

if __name__ == "__main__":
    main()
