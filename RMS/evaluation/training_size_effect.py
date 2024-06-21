''' Line plot with error bars for the effect
    of training set size on performance'''

import os
import os.path
import sys
import ipdb
import argparse

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use("PDF")
mpl.rcParams["pdf.fonttype"]=42
mpl.rcParams["agg.path.chunksize"]=10000

import matplotlib.pyplot as plt
plt.style.use('paper_alt.mplstyle')
import seaborn as sns

def execute(configs):
    df=pd.read_csv(configs["stats_path"],sep='\t')
    x_vect=[100,50,25,10,5,2,1]
    auprc_mean=df["auprc_mean"].values
    auprc_std=df["auprc_std"].values
    prec80_mean=df["prec80_mean"].values
    prec80_std=df["prec80_std"].values

    cm=1/2.54
    h=4
    w=6
    
    fig,ax1=plt.subplots(figsize=(w*cm,h*cm))

    color = 'tab:red'
    ax1.plot(x_vect,auprc_mean,color=color,label="AUPRC")
    ax1.fill_between(x_vect,auprc_mean-auprc_std,auprc_mean+auprc_std,alpha=0.2,color=color)
    ax1.set_xlabel("Training set size [%]")
    ax1.set_ylabel("AUPRC",color=color)
    ax1.set_ylim((0.2,0.6))
    ax1.grid(which="both",lw=0.1)    
    ax1.set_xticks([1,2,5,10,25,50,100])    
    ax1.tick_params(axis='y',labelcolor=color)
    
    ax2=ax1.twinx()

    color = 'tab:blue'    
    ax2.plot(x_vect,prec80_mean,color=color,label="Prec@80")
    ax2.fill_between(x_vect,prec80_mean-prec80_std,prec80_mean+prec80_std,alpha=0.2,color=color)
    ax2.set_ylabel("Precision @ 80 % Recall",color=color)
    ax2.set_ylim((15,40))
    ax2.tick_params(axis='y',labelcolor=color)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    plt.savefig(os.path.join(configs["plot_path"],"auprc_prec80_training_size.pdf"))
    plt.savefig(os.path.join(configs["plot_path"],"auprc_prec80_training_size.png"),dpi=400)    
    plt.clf()
    

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--stats_path", default="../../data/evaluation/time_point_based/eval_ef_training_size/task_results_with_error_bars.tsv",
                        help="Statistics to load")

    # Output paths
    parser.add_argument("--plot_path", default="../../data/plots/training_set_size",
                        help="Plot output path")

    # Arguments

    configs=vars(parser.parse_args())

    execute(configs)
