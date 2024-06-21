''' Plot bar and spider plots of interesting hexes in the
    hex plots'''

import os
import os.path
import sys
import ipdb
import argparse
import math
import textwrap

import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
import scipy.stats as sp_stats

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import plotly.express as px




def execute(configs):

    if configs["outcome_desc"]=="risk_rf_vs_vent": # Interesting hexes {240 (TR),160 (TL),144(BR),86(BL)}
        df_rf_scores=pd.read_parquet(os.path.join(configs["stats_path"],"{}_rf_MEDIAN.parquet".format(configs["outcome_desc"])))
        df_vent_scores=pd.read_parquet(os.path.join(configs["stats_path"],"{}_vent_MEDIAN.parquet".format(configs["outcome_desc"])))
        rf_scores_hexes=df_rf_scores["median_value"].values
        vent_scores_hexes=df_vent_scores["median_value"].values
        common_frame=pd.DataFrame({"rf_score": rf_scores_hexes, "vent_score": vent_scores_hexes,
                                   "x_coord": df_rf_scores["x_coord"].values,
                                   "y_coord": df_rf_scores["y_coord"].values})
        common_frame.reset_index(inplace=True)
        x=rf_scores_hexes
        y=vent_scores_hexes

    elif configs["outcome_desc"]=="risk_rf_vs_rexp": # Interesting Hexes {0 (TR) ,11 (BR),93 (TL), 144 (BL)}
        df_rf_scores=pd.read_parquet(os.path.join(configs["stats_path"],"{}_rf_MEDIAN.parquet".format(configs["outcome_desc"])))
        df_rexp_scores=pd.read_parquet(os.path.join(configs["stats_path"],"{}_rexp_MEDIAN.parquet".format(configs["outcome_desc"])))
        rf_scores_hexes=df_rf_scores["median_value"].values
        rexp_scores_hexes=df_rexp_scores["median_value"].values
        common_frame=pd.DataFrame({"rf_score": rf_scores_hexes, "rexp_score": rexp_scores_hexes,
                                   "x_coord": df_rf_scores["x_coord"].values,
                                   "y_coord": df_rf_scores["y_coord"].values})
        common_frame.reset_index(inplace=True)
        x=rf_scores_hexes
        y=rexp_scores_hexes

    elif configs["outcome_desc"]=="risk_rf_vs_ef": # Interesting Hexes {0 (TR) ,11 (BR),93 (TL), 144 (BL)}
        df_rf_scores=pd.read_parquet(os.path.join(configs["stats_path"],"{}_rf_MEDIAN.parquet".format(configs["outcome_desc"])))
        df_ef_scores=pd.read_parquet(os.path.join(configs["stats_path"],"{}_ef_MEDIAN.parquet".format(configs["outcome_desc"])))
        ef_scores_hexes=df_ef_scores["median_value"].values
        rf_scores_hexes=df_rf_scores["median_value"].values
        common_frame=pd.DataFrame({"ef_score": ef_scores_hexes, "rf_score": rf_scores_hexes,
                                   "x_coord": df_rf_scores["x_coord"].values,
                                   "y_coord": df_rf_scores["y_coord"].values})
        common_frame.reset_index(inplace=True)
        x=rf_scores_hexes
        y=ef_scores_hexes

    elif configs["outcome_desc"]=="risk_rf":
        df_rf_scores=pd.read_parquet(os.path.join(configs["stats_path"],"{}_rf_MEDIAN.parquet".format(configs["outcome_desc"])))

    elif configs["outcome_desc"]=="risk_ef":
        df_ef_scores=pd.read_parquet(os.path.join(configs["stats_path"],"{}_ef_MEDIAN.parquet".format(configs["outcome_desc"])))

    if "_vs_" in configs["outcome_desc"]:
        z = np.polyfit(x, y, 1)

        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x,y)

        y_hat = np.poly1d(z)(x)

        print("Creating scatter plot...")

        plt.plot(x,y,"+", ms=10, mec="k")
        plt.plot(x, y_hat, "r--", lw=1)
        text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"

    if configs["outcome_desc"]=="risk_rf_vs_vent":
        plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
                       fontsize=14, verticalalignment='top')        
        plt.xlabel("Respiratory failure median score in hex",fontsize=16)
        plt.ylabel("Ventilation need median score in hex",fontsize=16)
        plt.savefig(os.path.join(configs["plot_path"],"scatter_rf_vs_vent.png"),dpi=300)
        
    elif configs["outcome_desc"]=="risk_rf_vs_rexp":
        plt.gca().text(0.05, 0.15, text,transform=plt.gca().transAxes,
                       fontsize=14, verticalalignment='top')                
        plt.xlabel("Respiratory failure median score in hex",fontsize=16)
        plt.ylabel("Readiness to extubate median score in hex",fontsize=16)
        plt.savefig(os.path.join(configs["plot_path"],"scatter_rf_vs_rexp.png"),dpi=300)

    elif configs["outcome_desc"]=="risk_rf_vs_ef":
        plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
                       fontsize=14, verticalalignment='top')                
        plt.xlabel("Respiratory failure median score in hex",fontsize=16)
        plt.ylabel("Extubation failure median score in hex",fontsize=16)
        plt.savefig(os.path.join(configs["plot_path"],"scatter_rf_vs_ef.png"),dpi=300)
        
    plt.clf()

    # Now create a parallel coordinates plot with the distribution in the individual variables

    df_dict={}

    score_names=["ef","rf","status_rf","status_mort","status_early_ef_pred"]
    important_vars=score_names+configs["IMPORTANT_VARS"]    
    
    for var in important_vars:
        print("Loading variable: {}".format(var))

        df_var_median=pd.read_parquet(os.path.join(configs["stats_path"],"{}_{}_MEDIAN.parquet".format(configs["outcome_desc"],
                                                                                              var)))
        df_var_Q1=pd.read_parquet(os.path.join(configs["stats_path"],"{}_{}_Q1.parquet".format(configs["outcome_desc"],
                                                                                                 var)))
        df_var_Q3=pd.read_parquet(os.path.join(configs["stats_path"],"{}_{}_Q3.parquet".format(configs["outcome_desc"],
                                                                                               var)))        
        df_var_median.reset_index(inplace=True)
        df_var_Q1.reset_index(inplace=True)
        df_var_Q3.reset_index(inplace=True)
        
        #df_var=df_var[df_var.index.isin(configs["INTERESTING_HEXES"])]
        df_var_median.rename(columns={"index": "hex_name"},inplace=True)
        df_var_median.sort_values(by="hex_name",inplace=True)
        df_var_Q1.rename(columns={"index": "hex_name"},inplace=True)
        df_var_Q1.sort_values(by="hex_name",inplace=True)
        df_var_Q3.rename(columns={"index": "hex_name"},inplace=True)
        df_var_Q3.sort_values(by="hex_name",inplace=True)
        
        assert (df_var_median.x_coord==df_var_Q1.x_coord).all()

        var_values_median=df_var_median["median_value"].values
        var_values_Q1=df_var_Q1["q1_value"].values
        var_values_Q3=df_var_Q3["q3_value"].values        
            
        df_dict["{}_MEDIAN".format(var)]=var_values_median
        df_dict["{}_Q1".format(var)]=var_values_Q1
        df_dict["{}_Q3".format(var)]=var_values_Q3

    px_frame=pd.DataFrame(df_dict)
    int_frame=px_frame[px_frame.index.isin(configs["INTERESTING_HEXES"])]
    ipdb.set_trace()
    print("FOO")


if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--stats_path", default="../../data/plots/umap",
                        help="Path from where to load the summary statistics")

    # Output paths
    parser.add_argument("--plot_path", default="../../data/plots/umap",
                        help="Path where to output the spider and bar plots")

    # Arguments
    #parser.add_argument("--outcome_desc", default="risk_rf",
    #                    help="Input outcome which should be loaded")
    parser.add_argument("--outcome_desc", default="status_early_ef_pred",
                        help="Input outcome which should be loaded")
    #parser.add_argument("--outcome_desc", default="risk_ef",
    #                    help="Input outcome which should be loaded")    
    #parser.add_argument("--outcome_desc", default="risk_rf_vs_rexp",
    #                    help="Input outcome which should be loaded")
    #parser.add_argument("--outcome_desc", default="risk_rf_vs_ef",
    #                    help="Input outcome which should be loaded")      

    configs=vars(parser.parse_args())

    # RF (HEX=204)
    #configs["INTERESTING_HEXES"]=[204]
    #configs["INTERESTING_HEXES"]=[317]    

    # EF (HEX=41), High EF risk, ready-to-extubate
    #configs["INTERESTING_HEXES"]=[41]

    # Early EF pred (HEX=6,43,70)
    configs["INTERESTING_HEXES"]=[6,43,70]

    configs["IMPORTANT_VARS"]=["vm211", # Pressure support (INCLUDED)
                               "pm77", # Benzodiacepine (INCLUDED)
                               "pm39", # Norepinephrine (INCLUDED)
                               "vm58", # FiO2 (INCLUDED)
                               "pm80", # Propofol (INCLUDED)
                               "vm216", # MV spont servo (INCLUDED)
                               "vm318", # Atemfrequenz (INCLUDED)
                               "vm319", # RRsp m (INCLUDED)
                               "vm23", # Supplemental oxygen (INCLUDED)
                               "vm26", # GCS Motor (INCLUDED)
                               "vm20", # SpO2 (INCLUDED)
                               "vm140", # PaO2 (INCLUDED)
                               "pm83", # Insulin kurzwirksam (INCLUDED)
                               "vm309", # Supplemental FiO2 % (INCLUDED)
                               "vm62", # Spitzendruck (INCLUDED)
                               "vm293"] # PEEPs (INCLUDED)    

    labels_ef={"vm211": "Pressure sup.", # Pressure support
               "pm77": "Benzo.", # Benzodiacepine
               "pm39": "Noreph.", # Norepinephrine
               "vm58": "FiO2", # FiO2
               "pm80": "Propofol", # Propofol
               "vm216": "MV spont s..", # MV spont servo
               "vm318": "Vent RR", # Atemfrequenz
               "vm319": "RRsp m", # RRsp m
               "vm23": "Supp. Ox.", # Supplemental oxygen
               "pm83": "Insulin fast."} # Insulin kurzwirksam    

    labels_rf={"vm58": "FiO2", # FiO2
               "vm20": "SpO2", # SpO2
               "vm140": "PaO2", # PaO2
               "vm309": "Supp.FiO2%", # Supplemental FiO 2
               "pm69": "Loop diur.", # Loop diuretics
               "pm95": "Heparin", # Heparin
               "vm26": "GCS Motor", # GCS Motor
               "vm62": "Vent. peak p..", # Spitzendruck
               "vm23": "Supp ox.", # Supplementawl oxygen
               "pm80": "Propofol", # Propofol
               "vm293": "PEEPs"} # PEEPs

    labels_ef.update(labels_rf)
    
    configs["IMPORTANT_VARS_LABELS"]=labels_ef

    execute(configs)
