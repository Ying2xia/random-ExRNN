"""Plot simulation errors and application profiles from the scalar result files."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
LEVELS=np.array([.05,.10,.30,.50,.70,.90,.95])
METHODS=['Centralized ERNN','Host ERNN','ExtremileFed']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'published_results')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/figures')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(args.input/'main.csv')
    errors=['normal','t3','lognormal','contaminated']
    fig,axes=plt.subplots(4,3,figsize=(11,10),constrained_layout=True)
    for row,error in enumerate(errors):
        for col,scenario in enumerate([1,2,3]):
            ax=axes[row,col]
            setting=data[data.setting==f'scenario{scenario}_{error}']
            for method in METHODS:
                values=setting[setting.method==method][[f'extremile_rmse_tau_{x:.2f}' for x in LEVELS]]
                ax.errorbar(LEVELS,values.mean(),yerr=values.std(ddof=1)/np.sqrt(len(values)),marker='o',markersize=3,label=method)
            ax.set(title=f'Example {scenario}, {error}',xlabel='Extremile level',ylabel='RMSE',yscale='log')
            ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3)
    fig.savefig(args.output/'simulation_rmse.png',dpi=200)
    plt.close(fig)

    profiles=pd.read_csv(args.input/'real_profiles.csv')
    for dataset in ['fremtpl2','beijing']:
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
        for method,label in [('centralized','Centralized ExRNN'),('host','Host ExRNN'),('extremilefed','ExtremileFed')]:
            subset=profiles[(profiles.dataset==dataset)&(profiles.method==method)].sort_values('tau')
            axes[0].plot(subset.tau,subset.mean_prediction,marker='o',label=label)
            axes[1].plot(subset.tau,subset.mean_absolute_centralized_gap,marker='o',label=label)
        for ax in axes:ax.set_xlabel('Extremile level');ax.grid(alpha=.2)
        axes[0].set_ylabel('Mean fitted extremile')
        axes[1].set_ylabel('Mean absolute gap from centralized fit')
        axes[0].legend(fontsize=8)
        fig.savefig(args.output/(dataset+'_profiles.png'),dpi=200)
        plt.close(fig)
    print('Saved scalar-result plots under',args.output)


if __name__=='__main__':main()
