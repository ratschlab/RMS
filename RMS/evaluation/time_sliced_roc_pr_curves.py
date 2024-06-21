''' Plot a time-slice based ROC/PR plot using 
    a set of saved curves to the file-system'''

import argparse
import os
import os.path
import ipdb
import pickle
import sys

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use("PDF")
mpl.rcParams["pdf.fonttype"]=42
mpl.rcParams["agg.path.chunksize"]=10000

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

import seaborn as sns
current_palette=sns.color_palette()
import scipy
import sklearn.metrics as skmetrics

def execute(configs):

    #split_sets=[("temporal_1","random_1"),
    #           ("temporal_2","random_1"),
    #           ("temporal_3","random_1"),
    #           ("temporal_4","random_1"),
    #           ("temporal_5","random_1")]

    split_sets=[("temporal_1","random_1"),
                ("temporal_2","random_2"),
                ("temporal_3","random_3"),
                ("temporal_4","random_4"),
                ("temporal_5","random_5")]    

    with open(configs["result_path"],'rb') as fp:
        obj=pickle.load(fp)
        roc_curves=obj["roc_curves"]
        pr_curves=obj["pr_curves"]

        # Create ROC curve

        cm=1/2.54
        h=5.8
        w=6
        plt.figure(figsize=(w*cm,h*cm))
        
        ax=plt.gca()
        ax.set_rasterization_zorder(1)
        plt.text(0.07,0.20,"AUROC",color='k')
        #plt.text(0.07,0.10,"AUROC",color='k')        

        for database,label_key,curve_label,color_key,score_y in configs["CURVE_LABELS"]:

            try:
                curves=roc_curves[(database,label_key,("temporal_1","random_1"))]
            except KeyError:
                ipdb.set_trace()
                print("FOO")
                
            main_fpr=curves["fpr"]
            main_tpr=curves["tpr"]
            sort_arr=np.argsort(main_fpr)
            main_fpr=main_fpr[sort_arr]
            main_tpr=main_tpr[sort_arr]
            tprs=[]
            auroc_main=skmetrics.auc(main_fpr,main_tpr)
            auroc_scores_var=[]
            for var_split_bern,var_split_umc in split_sets:
                try:
                    var_curve=roc_curves[(database,label_key,(var_split_bern,var_split_umc))]
                except KeyError:
                    ipdb.set_trace()
                    print("FOO")
                    
                var_fpr=var_curve["fpr"]
                var_tpr=var_curve["tpr"]
                sort_arr=np.argsort(var_fpr)
                var_fpr=var_fpr[sort_arr]
                var_tpr=var_tpr[sort_arr]
                tprs.append(scipy.interp(main_fpr,var_fpr,var_tpr))
                auroc_scores_var.append(skmetrics.auc(var_fpr,var_tpr))
            std_tprs=np.std(tprs,axis=0)
            tprs_upper=np.minimum(main_tpr+std_tprs,1)
            tprs_lower=np.maximum(main_tpr-std_tprs,0)
            plt.plot(curves["fpr"],curves["tpr"],rasterized=True,zorder=0,color=color_key,label=curve_label)
            plt.fill_between(main_fpr,tprs_lower,tprs_upper,alpha=0.2,rasterized=True,zorder=0,color=color_key)
            auroc_stdev=np.std(auroc_scores_var)
            plt.text(0.07,score_y,r'{:.3f}$\pm${:.3f}'.format(auroc_main,auroc_stdev),color=color_key)

        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.plot([0,1],[0,1],color="grey",lw=0.5,linestyle="--",rasterized=True,zorder=0)

        # Add manual additional markers for the baselines (ROC curve)
        #plt.plot(0.309, 0.432, marker='o', label="Baseline Ready Ext Score")
        #plt.plot(0.824, 0.889, marker='x', label="Baseline Ready Ext Any Violation")
        
        plt.legend(loc="lower right")
        ax=plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect(1.0)
        ax.grid(which="both",lw=0.1)
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.tight_layout()
        plt.savefig(os.path.join(configs["plot_path"],"roc_main_{}.pdf".format(configs["plot_name"])))
        plt.savefig(os.path.join(configs["plot_path"],"roc_main_{}.png".format(configs["plot_name"])))
        plt.clf()

        # Create PR curve
        cm=1/2.54
        h=5.8
        w=6
        plt.figure(figsize=(w*cm,h*cm))
        
        ax=plt.gca()
        ax.set_rasterization_zorder(1)

        # Metric name
        plt.text(0.05,0.20,"AUPRC",color='k')
        #plt.text(0.05,0.15,"AUPRC",color='k')

        # Model name for external validation plots
        plt.text(0.54,0.98,"RMS-EF")
        #plt.text(0.54,0.98,"RMS-EF-lite")        
        
        for database,label_key,curve_label,color_key,score_y in configs["CURVE_LABELS"]:

            curves=pr_curves[(database,label_key,("temporal_1","random_1"))]
            main_rec=curves["recalls"]
            sort_arr=np.argsort(main_rec)
            main_rec=main_rec[sort_arr]
            precs=[]
            auprc_scores_var=[]
            for var_split_bern,var_split_umc in split_sets:
                var_curve=pr_curves[(database,label_key,(var_split_bern,var_split_umc))]
                var_rec=var_curve["recalls"]
                var_prec=var_curve["precs"]
                sort_arr=np.argsort(var_rec)
                var_rec=var_rec[sort_arr]
                var_prec=var_prec[sort_arr]
                precs.append(scipy.interp(main_rec,var_rec,var_prec))
                auprc_scores_var.append(skmetrics.auc(var_rec,var_prec))
            std_precs=np.std(precs,axis=0)
            mean_precs=np.mean(precs,axis=0)
            precs_upper=np.minimum(mean_precs+std_precs,1)
            precs_lower=np.maximum(mean_precs-std_precs,0)
            plt.plot(main_rec,mean_precs,rasterized=True,zorder=0,color=color_key,label=curve_label)
            plt.fill_between(main_rec,precs_lower,precs_upper,alpha=0.2,rasterized=True,zorder=0,color=color_key)
            auprc_stdev=np.std(auprc_scores_var)
            auprc_main=skmetrics.auc(main_rec,mean_precs)
            plt.text(0.05,score_y,r'{:.3f}$\pm${:.3f}'.format(auprc_main,auprc_stdev),color=color_key)
            print("PR score: {:.3f}".format(auprc_main))

        plt.xlim((0,1))
        plt.ylim((0,1))

        plt.legend(loc="upper right")
        plt.grid(alpha=0.25)
        ax=plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect(1.0)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(os.path.join(configs["plot_path"],"pr_main_{}.pdf".format(configs["plot_name"])))
        plt.savefig(os.path.join(configs["plot_path"],"pr_main_{}.png".format(configs["plot_name"])),dpi=400)
        plt.clf()        

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    #parser.add_argument("--result_path", default="../../data/evaluation/time_point_based/eval_ext_fail_internal/raw_results.pickle",
    #                    help="Path where result curves should be loaded")
    parser.add_argument("--result_path", default="../../data/evaluation/time_point_based/eval_ef_extval/raw_results.pickle",
                        help="Path where result curves should be loaded")
    #parser.add_argument("--result_path", default="../../data/evaluation/time_point_based/eval_eflite_extval/raw_results.pickle",
    #                    help="Path where result curves should be loaded")    
    #parser.add_argument("--result_path", default="../../data/evaluation/time_point_based/eval_vent_internal/raw_results.pickle",
    #                    help="Path where result curves should be loaded")
    #parser.add_argument("--result_path", default="../../data/evaluation/time_point_based/eval_rexp_internal/raw_results.pickle",
    #                    help="Path where result curves should be loaded")
    #parser.add_argument("--result_path", default="../../data/evaluation/time_point_based/eval_rf_internal/raw_results.pickle",
    #                    help="Path where result curves should be loaded")            

    # Output paths
    #parser.add_argument("--plot_path", default="../../data/plots/oxygenation_failure",
    #                    help="Path where plots should be saved")
    parser.add_argument("--plot_path", default="../../data/plots/extubation_failure",
                        help="Path where plots should be saved")    

    # Arguments
    parser.add_argument("--plot_name", default="EXTFAIL_extval", help="Name for plot")
    #parser.add_argument("--plot_name", default="EXTFAILlite_extval", help="Name for plot")    
    #parser.add_argument("--plot_name", default="EXTFAIL_main", help="Name for plot")
    #parser.add_argument("--plot_name", default="VENT_main", help="Name for plot")
    #parser.add_argument("--plot_name", default="REXP_main", help="Name for plot")            
    #parser.add_argument("--plot_name", default="EXTFAIL_internal", help="Name for plot")
    #parser.add_argument("--plot_name", default="RESPFAIL_internal", help="Name for plot")
    #parser.add_argument("--plot_name", default="RESPFAIL_main",help="Name for plot")
    #parser.add_argument("--plot_name", default="RESPFAIL_extval", help="Name for plot")    
    #parser.add_argument("--plot_name", default="RESPFAIL_retrain", help="Name for plot")
    #parser.add_argument("--plot_name", default="RESPFAIL_val", help="Name for plot")  
    
    configs=vars(parser.parse_args())

    # Curves that should be loaded
    #configs["CURVE_LABELS"]=[("umcdb","Label_ExtubationFailureSimple_retrain_compact_lightgbm","random_1","EF (Retrain)","C4",0.20),
    #                         ("umcdb","Label_ExtubationFailureSimple_retrain_compact_drop_3P_lightgbm","random_1","EF (Retrain, drop 3P)","C5",0.15),
    #                         ("umcdb","Label_ExtubationFailureSimple_retrain_compact_drop_5P_lightgbm","random_1","EF (Retrain, drop 5P)","C6",0.10),                             
    #                         ("umcdb","Label_ExtubationFailureSimple_retrain_compact_no_pharma_lightgbm","random_1","EF (Retrain, drop pharma)","C7",0.05)]

    # configs["CURVE_LABELS"]=[("umcdb","Label_WorseStateFromZeroOrOne0To24Hours_retrain_compact_lightgbm","random_1","EF (Retrain)","C4",0.20),
    #                          ("umcdb","Label_WorseStateFromZeroOrOne0To24Hours_retrain_compact_drop_3P_lightgbm","random_1","EF (Retrain, drop 3P)","C5",0.15),
    #                          ("umcdb","Label_WorseStateFromZeroOrOne0To24Hours_retrain_compact_drop_5P_lightgbm","random_1","EF (Retrain, drop 5P)","C6",0.10),
    #                          ("umcdb","Label_WorseStateFromZeroOrOne0To24Hours_retrain_compact_no_pharma_lightgbm","random_1","EF (Retrain, drop pharma)","C7",0.05)]

    # configs["CURVE_LABELS"]=[("umcdb","Label_WorseStateFromZeroOrOne0To24Hours_val_compact_lightgbm","random_1","EF (Validation)","C0",0.20),
    #                          ("umcdb","Label_WorseStateFromZeroOrOne0To24Hours_val_compact_drop_3P_lightgbm","random_1","EF (Validation, drop 3P)","C1",0.15),
    #                          ("umcdb","Label_WorseStateFromZeroOrOne0To24Hours_val_compact_drop_5P_lightgbm","random_1","EF (Validation, drop 5P)","C2",0.10),                                
    #                          ("umcdb","Label_WorseStateFromZeroOrOne0To24Hours_val_compact_no_pharma_lightgbm","random_1","EF (Val, drop pharma)","C3",0.05)]

    # configs["CURVE_LABELS"]=[("hirid","Label_WorseStateFromZeroOrOne0To24Hours_internal_compact_lightgbm","temporal_1","EF (Internal)","C0",0.20),
    #                          ("hirid","Label_WorseStateFromZeroOrOne0To24Hours_internal_compact_drop_3P_lightgbm","temporal_1","EF (Internal, drop 3P)","C1",0.15),
    #                          ("hirid","Label_WorseStateFromZeroOrOne0To24Hours_internal_compact_drop_5P_lightgbm","temporal_1","EF (Internal, drop 5P)","C2",0.10),                                
    #                          ("hirid","Label_WorseStateFromZeroOrOne0To24Hours_internal_compact_no_pharma_lightgbm","temporal_1","EF (Internal, drop pharma)","C3",0.05)]
 
    # configs["CURVE_LABELS"]=[("hirid","Label_WorseStateFromZeroOrOne0To24Hours_internal_compact_lightgbm","temporal_1","EF (Internal)","C0",0.15),
    #                          ("umcdb","Label_WorseStateFromZeroOrOne0To24Hours_val_compact_no_pharma_lightgbm","random_1","EF (Val, drop pharma)","C1",0.10),
    #                          ("umcdb","Label_WorseStateFromZeroOrOne0To24Hours_retrain_compact_lightgbm","random_1","EF (Retrain)","C2",0.05)]
 
    #configs["CURVE_LABELS"]=[("hirid","Label_WorseStateFromZeroOrOne0To24Hours_internal_rmsRF_lightgbm","RMS-RF","C0",0.15), 
    #                         ("hirid","Label_WorseStateFromZeroOrOne0To24Hours_clinical_baseline_tree","Decision Tree","C1",0.10),
    #                         ("hirid","Label_WorseStateFromZeroOrOne0To24Hours_one_minus_spo2_fio2_ratio","SpO2/FiO2","C2",0.05)]

    # configs["CURVE_LABELS"]=[("hirid","Label_ExtubationFailureSimple_internal_compact_lightgbm","respEWS","C0",0.15),
    #                          ("hirid","Label_ExtubationFailureSimple_internal_compact_drop_pharma_lightgbm","respEWS-lite","C1",0.10),  
    #                          ("hirid","Label_ExtubationFailureSimple_internal_compact_lightgbm_custom_threshold","Baseline (Ready Ext. violation score)","C2",0.05)]

    #configs["CURVE_LABELS"]=[("hirid","Label_ExtubationFailureSimple_internal_rmsEF_lightgbm","RMS-EF","C0",0.10),
    #                         ("hirid","Label_ExtubationFailureSimple_internal_rmsEF_lightgbm_custom_threshold","REXT status score","C2",0.05)]

    #configs["CURVE_LABELS"]=[("hirid","Label_Ventilation0To24Hours_internal_rmsVENT_lightgbm","RMS-VENT","C0",0.05)]
 
    #configs["CURVE_LABELS"]=[("hirid","Label_ReadyExtubate0To24Hours_internal_rmsREXP_lightgbm","RMS-REXT","C0",0.05)]   

    configs["CURVE_LABELS"]=[("hirid","Label_ExtubationFailureSimple_internal_rmsEF_lightgbm","HiRID->HiRID","C9",0.15),
                            ("umcdb","Label_ExtubationFailureSimple_val_rmsEF_lightgbm","HiRID->UMCDB","C1",0.10),
                            ("umcdb","Label_ExtubationFailureSimple_retrain_rmsEF_lightgbm","UMCDB->UMCDB","C8",0.05)]

    #configs["CURVE_LABELS"]=[("hirid","Label_ExtubationFailureSimple_internal_rmsEFlite_lightgbm","HiRID->HiRID","C9",0.15),
    #                         ("umcdb","Label_ExtubationFailureSimple_val_compact_UMCDB_20var_prefix_no_pharma_lightgbm","HiRID->UMCDB","C1",0.10),
    #                         ("umcdb","Label_ExtubationFailureSimple_retrain_rmsEFlite_lightgbm","UMCDB->UMCDB","C8",0.05)]    
    
    execute(configs)
