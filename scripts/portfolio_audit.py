"""Offline, artifact-only audit. Never imports project or writes to input tree."""
import argparse
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def metrics(r, periods=52):
    r = np.asarray(r, dtype=float)
    if len(r) < 2 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError('Invalid return series')
    equity = np.r_[1., np.cumprod(1+r)]
    dd = equity / np.maximum.accumulate(equity) - 1
    vol = r.std(ddof=1) * np.sqrt(periods)
    downside = np.sqrt(np.mean(np.minimum(r,0)**2)) * np.sqrt(periods)
    cagr = equity[-1]**(periods/len(r))-1
    return dict(n=len(r), cumulative=equity[-1]-1, annualized_geometric=cagr,
                volatility=vol, sharpe=r.mean()*periods/vol if vol else np.nan,
                sortino=r.mean()*periods/downside if downside else np.nan,
                drawdown=dd.min(), calmar=cagr/abs(dd.min()) if dd.min()<0 else np.nan)


def read(root, name):
    return pd.read_csv(root/'data'/name)


def run(root, out):
    root, out = root.resolve(), out.resolve()
    if out == root or root in out.parents:
        raise ValueError('Output must be outside input repository')
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    used=[]; rows=[]
    def load(name):
        used.append(name); return read(root,name)
    def add(group,name,s,exposure=None,turnover=None):
        s=s.sort_index(); assert s.index.is_unique
        result=dict(group=group, variant=name,start=str(s.index.min().date()),end=str(s.index.max().date()),**metrics(s))
        if exposure is not None: result.update(avg_exposure=exposure.mean(),max_exposure=exposure.max())
        if turnover is not None: result['avg_turnover']=turnover.mean()
        rows.append(result)
    def charts(wide,label):
        eq=(1+wide).cumprod(); dd=eq/eq.cummax().clip(lower=1)-1
        for name,df,y in [('equity',eq,'Growth of 1'),('drawdown',dd*100,'Drawdown (%)'),('rolling_vol',wide.rolling(26).std()*np.sqrt(52)*100,'26-observation volatility (%)')]:
            ax=df.plot(figsize=(10,5),title=f'{label}: {name.replace("_"," ")}'); ax.set_ylabel(y); ax.grid(alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(out/f'{label}_{name}.png',dpi=150); plt.close(ax.figure)
    h=load('growth_head_to_head_daily_returns.csv'); h.date=pd.to_datetime(h.date)
    wide=h.pivot(index='date',columns='candidate',values='return').dropna()
    for n in wide:
        sub=h[h.candidate==n].set_index('date').reindex(wide.index)
        add('aligned_head_to_head',n,wide[n],sub.target_exposure,sub.turnover)
    charts(wide,'aligned_head_to_head')
    f=load('growth_final_selection_daily_returns.csv'); f.date=pd.to_datetime(f.date)
    for window,g in f.groupby('window_start'):
        w=g.pivot(index='date',columns='candidate',values='candidate_return').dropna()
        for n in w:
            sub=g[g.candidate==n].set_index('date').reindex(w.index)
            add('reconstruction_'+window,n,w[n],sub.candidate_exposure,sub.candidate_turnover)
        if window=='2008-01-01': charts(w,'long_reconstruction')
    result=pd.DataFrame(rows); result.to_csv(out/'portfolio_metrics.csv',index=False)
    a=result[result.group=='aligned_head_to_head']; fig,ax=plt.subplots(figsize=(9,5)); ax.scatter(a.volatility*100,a.annualized_geometric*100)
    for _,row in a.iterrows(): ax.annotate(row.variant,(row.volatility*100,row.annualized_geometric*100),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Annualized volatility (%)',ylabel='Annualized geometric return (%)',title='Aligned historical research — not an efficient frontier'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/'risk_return.png',dpi=150); plt.close(fig)
    p=load('full_cscv_results.csv'); p.groupby('S').overfit_fold.agg(['size','mean']).to_csv(out/'cscv_summary.csv')
    for n in ['final_champion_backtest_daily_returns.csv','growth_champion_canonical_daily.csv','growth_official_paper_performance.csv']:
        d=load(n); cols=[c for c in ['realized_portfolio_return_1d','realized_portfolio_return_5d','gross_daily_return','net_daily_return','estimated_net_daily_return'] if c in d]; print(n,len(d),{c:float((1+d[c]).prod()-1) for c in cols})
    (out/'source_hashes.txt').write_text('\n'.join(f'{hashlib.sha256((root/"data"/n).read_bytes()).hexdigest()}  data/{n}' for n in sorted(set(used)))+'\n')
    print(result.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); run(a.root,a.out)
