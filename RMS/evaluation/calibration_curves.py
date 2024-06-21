''' Plot calibration curves based on data stored on file-system'''

import os
import os.path
import pickle
import ipdb
import argparse

import numpy as np
import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

def execute(configs):

    cm=1/2.54
    h=5.8
    w=6
    plt.figure(figsize=(w*cm,h*cm))            

    with open(configs["model_cal_dict"],'rb') as fp:
        obj=pickle.load(fp)
        plt.plot(obj["perfect_cal_x"], obj["perfect_cal_y"], 'k:')
        
        #plt.plot(obj["raw_x"],obj["raw_y"],'^--',label="RMS-EF",color="C0")
        plt.plot(obj["raw_x"],obj["raw_y"],'^--',label="RMS-RF",color="C0")
        #plt.plot(obj["raw_x"],obj["raw_y"],'^--',label="RMS-VENT",color="C0")
        #plt.plot(obj["raw_x"],obj["raw_y"],'^--',label="RMS-REXT",color="C0")        
        
 #       plt.plot(obj["iso_x"],obj["iso_y"],'s-',label="Compact (Cal.) {:.3f} ({:.3f})".format(obj["iso_gini_mean"],obj["iso_gini_std"]),color="C1")
        plt.fill_between(obj["raw_x"],obj["raw_fill_min"],obj["raw_fill_max"],color="C0",alpha=0.2)
        plt.text(0.6,0.15,r'{:.3f}$\pm${:.3f}'.format(obj["raw_brier_mean"],obj["raw_brier_std"]),color="C0")
#        plt.fill_between(obj["iso_x"],obj["iso_fill_min"],obj["iso_fill_max"],color="C1",alpha=0.2)

    # with open(configs["baseline_cal_dict"],'rb') as fp:
    #     obj=pickle.load(fp)
    #     plt.plot(obj["raw_x"],obj["raw_y"],'^--',label="REXT status score",color="C2")
    #     plt.fill_between(obj["raw_x"],obj["raw_fill_min"],obj["raw_fill_max"],color="C2",alpha=0.2)
    #     plt.text(0.6,0.05,"{:.3f}$\pm${:.3f}".format(obj["raw_brier_mean"],obj["raw_brier_std"]),color="C2")

    with open(configs["tree_cal_dict"],'rb') as fp:
        obj=pickle.load(fp)
        plt.plot(obj["raw_x"],obj["raw_y"],'^--',label="Decision Tree",color="C1")
        plt.fill_between(obj["raw_x"],obj["raw_fill_min"],obj["raw_fill_max"],color="C1",alpha=0.2)
        plt.text(0.6,0.10,"{:.3f}$\pm${:.3f}".format(obj["raw_brier_mean"],obj["raw_brier_std"]),color="C1")

    with open(configs["sf_ratio_cal_dict"],'rb') as fp:
        obj=pickle.load(fp)
        plt.plot(obj["raw_x"],obj["raw_y"],'^--',label="SpO2/FiO2",color="C2")
        plt.fill_between(obj["raw_x"],obj["raw_fill_min"],obj["raw_fill_max"],color="C2",alpha=0.2)
        plt.text(0.6,0.05,"{:.3f}$\pm${:.3f}".format(obj["raw_brier_mean"],obj["raw_brier_std"]),color="C2")

    #plt.ylabel("Observed frequency of ext. failure")
    plt.ylabel("Observed frequency of resp. failure")
    #plt.ylabel("Observed frequency of vent. onset")
    #plt.ylabel("Observed frequency of readiness to ext.")        
    
    plt.xlabel("Mean risk score in bin")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left",facecolor="white")
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    plt.text(0.6,0.20,"Brier score",color='k')
    ax=plt.gca()
    ax.set_aspect(1.0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(configs["plot_path"],"cal_curve_risk_OXYFAIL.pdf"))
    plt.savefig(os.path.join(configs["plot_path"],"cal_curve_risk_OXYFAIL.png"),dpi=400)
    #plt.savefig(os.path.join(configs["plot_path"],"cal_curve_risk_VENT.pdf"))
    #plt.savefig(os.path.join(configs["plot_path"],"cal_curve_risk_VENT.png"),dpi=400)
    #plt.savefig(os.path.join(configs["plot_path"],"cal_curve_risk_REXT.pdf"))
    #plt.savefig(os.path.join(configs["plot_path"],"cal_curve_risk_REXT.png"),dpi=400)    
    #plt.savefig(os.path.join(configs["plot_path"],"cal_curve_risk_EXTFAIL.pdf"))
    #plt.savefig(os.path.join(configs["plot_path"],"cal_curve_risk_EXTFAIL.png"),dpi=400)
    
    plt.clf()

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    #parser.add_argument("--model_cal_dict", default="../../data/evaluation/calibration/Label_ExtubationFailureSimple_lightgbm.pickle",
    #                    help="Calibration curve data")
    parser.add_argument("--model_cal_dict", default="../../data/evaluation/calibration/Label_WorseStateFromZeroOrOne0To24Hours_lightgbm.pickle",
                        help="Calibration curve data")
    #parser.add_argument("--model_cal_dict", default="../../data/evaluation/calibration/Label_Ventilation0To24Hours_lightgbm.pickle",
    #                    help="Calibration curve data")
    #parser.add_argument("--model_cal_dict", default="../../data/evaluation/calibration/Label_ReadyExtubate0To24Hours_lightgbm.pickle",
    #                    help="Calibration curve data")        
    
    parser.add_argument("--baseline_cal_dict", default="../../data/evaluation/calibration/Label_ExtubationFailureSimple_baseline.pickle",
                         help="Calibration curve data")
    parser.add_argument("--sf_ratio_cal_dict", default="../../data/evaluation/calibration/Label_WorseStateFromZeroOrOne0To24Hours_sf_ratio_baseline.pickle",
                         help="Calibration curve data")
    parser.add_argument("--tree_cal_dict", default="../../data/evaluation/calibration/Label_WorseStateFromZeroOrOne0To24Hours_tree.pickle",
                         help="Calibration curve data")                

    # Output paths
    parser.add_argument("--plot_path", default="../../data/plots/calibration")

    # Arguments

    configs=vars(parser.parse_args())

    execute(configs)
