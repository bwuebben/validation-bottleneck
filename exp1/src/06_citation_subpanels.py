"""Test (b) citation control across model subpanels (added 2026-07-25).

Mirrors 04_score.py's test-(b) panel construction and clustered OLS verbatim, then reruns it on
three nested model sets, to report how the log-citations control behaves as independent training
lineages enter the pool. The five-model column reproduces results_b.txt (fut t = -1.75 vs -1.74,
log_cites t = +0.45 vs +0.44; the last digit differs only because this script uses pinv where
04_score uses inv).

Output: derived/citation_subpanels.csv
"""
import datetime as dt, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl
SRC = Path("/Users/bwuebben/git/papers_ai_1/llm_alpha_discovery/exp1/src")
DERIVED = SRC.parent / "derived"
sys.path.insert(0, str(SRC))
from _dp import get_dp
MATCHED = {"exact", "close"}

def ols_clustered(y, X, clusters):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    idxs = defaultdict(list)
    for i, c in enumerate(clusters): idxs[c].append(i)
    for c, idx in idxs.items():
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.maximum(np.diag(V), 0))

matches = {}
for line in (DERIVED/"matches.jsonl").read_text().splitlines():
    r = json.loads(line)
    for m in r["matches"]:
        pid = f'{r["file"].split("/")[1]}|{Path(r["file"]).stem}|{m["index"]}'
        matches[pid] = {"match": m.get("match"), "strength": m.get("strength","none")}
proposals = [json.loads(l) for l in (DERIVED/"proposals.jsonl").read_text().splitlines()]
print(f"{len(proposals)} proposals, {len(matches)} judged")

dp = get_dp()
pred = dp.signals.open_asset_pricing_doc().filter(pl.col("Cat.Signal")=="Predictor")
meta={}
for r in pred.iter_rows(named=True):
    try: yr=int(r["Year"])
    except (TypeError,ValueError): yr=None
    try: cites=float(r.get("GScholarCites202509") or 0)
    except (TypeError,ValueError): cites=0.0
    meta[r["Acronym"]]={"year":yr,"cites":cites}
ls = dp.factors.open_asset_pricing("op").filter(pl.col("port")=="LS").select(["signalname","date","ret"])
cache=defaultdict(list)
for r in ls.iter_rows(named=True): cache[r["signalname"]].append((r["date"],float(r["ret"])))
def ann_mean(sig,start=None,end=None):
    xs=[x for d,x in cache.get(sig,[]) if (start is None or d>=start) and (end is None or d<=end)]
    return None if len(xs)<12 else float(np.mean(xs))*12

counts=defaultdict(int)
for p in proposals:
    m=matches.get(p["proposal_id"])
    if m and m["strength"] in MATCHED and m["match"] in meta:
        counts[(p["model"],p["vintage"],m["match"])]+=1
allm=sorted({p["model"] for p in proposals})
print("models:",allm)
Xcols=["fut","pre","pre_missing","log_cites","pub_year"]
def run(models,label):
    panel=[]
    for model in models:
        for v in sorted({p["vintage"] for p in proposals}):
            for acro,mm in meta.items():
                if not mm["year"] or mm["year"]<=v: continue
                fut=ann_mean(acro,start=dt.date(v,2,1),end=dt.date(2024,12,31))
                if fut is None: continue
                pre_r=ann_mean(acro,end=dt.date(v-1,12,31))
                panel.append({"model":model,"vintage":v,"acronym":acro,
                    "count":counts.get((model,v,acro),0),"fut":fut,
                    "pre":pre_r if pre_r is not None else 0.0,
                    "pre_missing":1.0 if pre_r is None else 0.0,
                    "log_cites":np.log1p(mm["cites"]),"pub_year":float(mm["year"])})
    fe=sorted({(r["model"],r["vintage"]) for r in panel}); fi={k:i for i,k in enumerate(fe)}
    Xb=np.array([[r[c] for c in Xcols] for r in panel])
    FE=np.zeros((len(panel),len(fe)))
    for i,r in enumerate(panel): FE[i,fi[(r["model"],r["vintage"])]]=1.0
    X=np.hstack([Xb,FE]); y=np.array([float(r["count"]) for r in panel])
    beta,se=ols_clustered(y,X,[r["acronym"] for r in panel])
    print(f"\n--- {label} (N={len(y)}, FE={len(fe)}) ---")
    out={"panel":label,"n_models":len(models),"N":len(y),"fe_cells":len(fe)}
    for j,c in enumerate(Xcols):
        t=beta[j]/se[j] if se[j]>0 else float("nan")
        print(f"   {c:12s} beta={beta[j]:+.5f} t={t:+.2f}")
        out[f"{c}_beta"]=round(float(beta[j]),5); out[f"{c}_t"]=round(float(t),2)
    return out
rows=[]
for models,label in [([m for m in allm if m.startswith("gpt-")],"openai_3"),
                     ([m for m in allm if not m.startswith("claude")],"four_pre_claude"),
                     (allm,"all_five")]:
    rows.append(run(models,label))
pl.DataFrame(rows).write_csv(DERIVED/"citation_subpanels.csv")
print("\nwrote derived/citation_subpanels.csv")
