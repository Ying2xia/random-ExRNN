"""Recompute means and Monte Carlo standard errors from scalar reference CSVs."""
from pathlib import Path
import argparse
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def summarize(frame, groups):
    numeric=frame.select_dtypes(include='number').columns.difference(groups+['replication','depth','clients'])
    rows=[]
    for key,part in frame.groupby(groups,dropna=False,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        for metric in numeric:
            values=part[metric].dropna()
            if len(values)==0:continue
            rows.append({**dict(zip(groups,key)),'metric':metric,'n':len(values),
                         'mean':values.mean(),'mcse':values.std(ddof=1)/(len(values)**.5) if len(values)>1 else None})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'published_results')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/summaries')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for stem,groups in [('main',['setting','method']),('supplementary',['setting','variant','method'])]:
        frame=pd.read_csv(args.input/(stem+'.csv'))
        result=summarize(frame,groups)
        result.to_csv(args.output/(stem+'_summary.csv'),index=False)
        print(stem,len(frame),'input rows;',len(result),'summary rows')


if __name__=='__main__':main()
