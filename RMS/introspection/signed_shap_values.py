''' An analysis of signed SHAP values in the test set of a split
    for individual features, using a scatter plot of feature value 
    vs. SHAP value for the most relevant features'''

import csv
import os.path
import os
import ipdb
import argparse
import glob
import sys
import random
import pickle

import numpy as np
import numpy.random as nprand
import pandas as pd
import matplotlib as mpl
mpl.use("PDF")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")

def execute(configs):
    random.seed(2023)
    nprand.seed(2023)
    
    name_dict=np.load(configs["mid_dict"],allow_pickle=True).item()

    backup_dict={"vm202": "OUTDialysis/c", "RelDatetime": "Time since admission", "vm201": "Chronic kidney disease?",
                 "static_APACHEPatGroup": "APACHE Patient Group", "static_Age": "Patient age", "static_Sex": "Patient gender", "vm212": "vm212",
                 "vm224": "Spontanatmung", "vm275": "Peritoneal dialysis", "vm253": "M-kr arm li", "vm204": "Hematocrit", "vm200": "vm200", "vm255": "M-Kr Bein li", "vm214": "I:E (s)",
                 "vm215": "MV Exp", "vm309": "Supplemental FiO2 %", "pm290": "Abführend", "vm249": "PubReaktre", "vm240": "Woher?", "vm227": "Sekret Konsistenz",
                 "vm216": "MV Spont. servo","vm318": "Atemfrequenz",
                 "vm306": "Ventilator mode group", "vm308": "Ventilator mode mode", "vm307": "Ventilator mode subgroup","vm223": "Extubation time-point",
                 "vm313": "Tracheotomy state", "vm319": "RRsp/m", "vm312": "Intubation state", "vm315": "TVs", "vm293": "PEEP(s)", "vm211": "Pressure support", 
                 "vm226": "Sekret menge", "vm317": "Urine culture", "vm251": "NRS Ans 0-10", "vm256": "M-Kr Bein re","vm242": "Gehen 400m", "vm314": "TV(m)",
                 "fio2estimated": "FiO2 estimate","ventstate": "Ventilation state", "Age": "Patient age", "vm259": "PeriphHandLi", "Sex": "Gender",
                 "PatGroup": "Patient group"}

    static_encode_dict= {"Sex": {"M": 0, "F": 1}}

    vlabel_dict={"vm211_time_to_last_ms": "Pressure support \n (Time to last meas.)",
                 "pm77_iqr_H10": "Benzodiacepine (IQR dose \n short-term horizon)",
                 "pm39_iqr_H156": "Norepinephrine (IQR dose \n longest-term horizon)",
                 "vm58_iqr_H156": "FiO2 (IQR \n longest-term horizon)",
                 "pm80_time_to_last_ms": "Propofol \n (Time to last dose)",
                 "vm216_median_H10": "MV Spont. servo \n(Median short-term horizon)",
                 "vm318_median_H10": "Ventilator RR \n(Median short-term horizon)",
                 "vm319_meas_density_H156": "RRsp(m) Meas. freq.\n (longest-term horizon)",
                 "vm23_time_to_last_ms": "Supplemental oxygen, \n (Time to last dose)",
                 "pm83_iqr_H62": "Insulin fast acting (IQR \n long-term horizon)",
                 "vm58_median_H156": "FiO2 (Median \n longest-term horizon)",
                 "vm20_instable_l1_8h": "SpO2 (Fraction of \n last 8h [90,94] %)",
                 "plain_vm140": "Current PaO2 value",
                 "plain_vm309": "Current Supp. FiO2 %",
                 "vm62_time_to_last_ms": "Vent. peak pressure\n(Time to last meas.)",
                 "vm23_median_H25": "Supplemental oxygen\n (Median mid-term horizon)",
                 "vm293_median_H156": "PEEP(s) (Median \n longest-term horizon)",
                 "vm26_time_to_last_ms": "GCS Motor. \n(Time to last meas.)"}

    # Features that should be plotted
    feats_to_analyze=configs["FEATS_TO_ANALYZE"]

    df_static=pd.read_hdf(configs["static_path"], mode='r')        
    static_cols=df_static.columns.values.tolist()
    kept_static_cols=list(filter(lambda col: col in feats_to_analyze, static_cols))
    kept_pred_cols=["AbsDatetime","PredScore"]+list(map(lambda col: "RawShap_"+col, feats_to_analyze))
    kept_feat_cols=["AbsDatetime"]+list(filter(lambda col: col not in kept_static_cols, feats_to_analyze))

    with open(configs["split_path"],'rb') as fp:
        splits=pickle.load(fp)

    acc_dict={}

    for split in configs["SPLITS"]:
        print("Analyzing split: {}".format(split))
        predfs=glob.glob(os.path.join(configs["pred_path"], split, configs["model_config"], "batch_*.h5"))
        print("Number of batches: {}".format(len(predfs)))
        for fpath in sorted(predfs):
            batch_id=int(fpath.split('/')[-1].split(".")[0][6:])

            # No test IDs in here...
            if batch_id<90:
                continue
            
            featpath=os.path.join(configs["feat_path"], split, "batch_{}.h5".format(batch_id))
            with pd.HDFStore(fpath,'r') as hstore:
                all_pids=list(map(lambda item: int(item[2:]), list(hstore.keys())))
                print("Number of PIDs in batch {}: {}".format(batch_id, len(all_pids)))

            test_pids=splits[split]["test"]
            all_test_pids=list(set(all_pids).intersection(set(test_pids)))

            if len(all_test_pids)==0:
                continue

            df_feat_batch=pd.read_hdf(featpath,"/X",mode='r')

            if configs["small_sample"]:
                random.shuffle(all_test_pids)
                all_test_pids=all_test_pids[:100]
                
            for pid in all_test_pids:
                df_static_pid=df_static[df_static["PatientID"]==pid]
                df_pred=pd.read_hdf(fpath,"/p{}".format(pid),mode='r')[kept_pred_cols]
                df_feat=df_feat_batch[df_feat_batch.PatientID==pid][kept_feat_cols]
                df_merged=df_pred.merge(df_feat,how="inner",on=["AbsDatetime"])
                df_merged=df_merged[df_merged["PredScore"].notnull()]
                
                for col in kept_static_cols:
                    empty_arr=np.zeros(df_merged.shape[0])
                    fill_val=df_static_pid[col].iloc[0]
                    if col in static_encode_dict:
                        fill_val=static_encode_dict[col][fill_val]
                    empty_arr[:]=fill_val
                    df_merged[col]=empty_arr
                
                for feat in feats_to_analyze:
                    if feat+"_val" not in acc_dict:
                        acc_dict[feat+"_val"]=[]
                        acc_dict[feat+"_SHAP"]=[]
                    acc_dict[feat+"_val"].extend(list(df_merged[feat]))
                    acc_dict[feat+"_SHAP"].extend(list(df_merged["RawShap_"+feat]))
                    
            if configs["debug_mode"]:
                break

    print("Creating figures")

    for feat in feats_to_analyze:
        x_arr=np.array(acc_dict[feat+"_val"])
        y_arr=np.array(acc_dict[feat+"_SHAP"])
        idx=nprand.choice(np.arange(len(x_arr)), configs["rsample"], replace=False)
        x_arr=x_arr[idx]
        y_arr=y_arr[idx]

        h=sns.jointplot(x=x_arr,y=y_arr,kind="reg",height=8,joint_kws={"lowess": True, "marker": "x","scatter_kws": {"s": 1}, "line_kws": {"color": "gold"}})
        #h=sns.jointplot(x=x_arr,y=y_arr,kind="reg",joint_kws={"order": 3, "marker": "x","scatter_kws": {"s": 1}, "line_kws": {"color": "gold"}})        

        vlabel=vlabel_dict[feat]

        h.set_axis_labels(vlabel, 'SHAP value')
        plt.tight_layout()
        #plt.savefig(os.path.join(configs["plot_path"], "{}.pdf".format(feat)))
        plt.savefig(os.path.join(configs["plot_path"], "{}.png".format(feat)),dpi=300)
        plt.clf()

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--pred_path", default="../../data/predictions/reduced",
                        help="Path from where to load the predictions")
    parser.add_argument("--feat_path", default="../../data/ml_input/hirid2_features/reduced",
                        help="Path from where to load feature values")
    parser.add_argument("--mid_dict", default="../../data/misc/mid2string_v6.npy")
    parser.add_argument("--static_path", default="../../data/imputed/noimpute_hirid2/reduced/temporal_1/static.h5")
    parser.add_argument("--split_path", default="../../data/exp_design/temp_splits_hirid2.pickle", help="Split descriptor")

    # Output paths
    #parser.add_argument("--plot_path", default="../../data/plots/oxygenation_failure/introspection",
    #                    help="Plotting folder")
    parser.add_argument("--plot_path", default="../../data/plots/extubation_failure/introspection",
                         help="Plotting folder")    

    # Arguments
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Process one batch")
    parser.add_argument("--rsample", type=int, default=50000, help="Random samples to draw before plotting")
    parser.add_argument("--small_sample", default=False, action="store_true", help="Small sample?")
    
    #parser.add_argument("--model_config", default="Label_WorseStateFromZeroOrOne0To24Hours_internal_rmsRF_lightgbm", help="Model to analyze")
    parser.add_argument("--model_config", default="Label_ExtubationFailureSimple_internal_rmsEF_lightgbm", help="Model to analyze")

    configs=vars(parser.parse_args())

    configs["SPLITS"]=["temporal_1"]

    # Respiratory failure
    #configs["FEATS_TO_ANALYZE"]=["vm58_median_H156",
    #                             "vm20_instable_l1_8h",
    #                             "plain_vm140",
    #                             "plain_vm309",
    #                             "vm62_time_to_last_ms",
    #                             "vm23_median_H25",
    #                             "vm293_median_H156",
    #                             "vm211_time_to_last_ms",
    #                             "vm26_time_to_last_ms",
    #                             "vm318_median_H10"]
    
    # Extubation failure
    configs["FEATS_TO_ANALYZE"]=["vm211_time_to_last_ms",
                                 "pm77_iqr_H10",
                                 "pm39_iqr_H156",
                                 "vm58_iqr_H156",
                                 "pm80_time_to_last_ms",
                                 "vm216_median_H10",
                                 "vm318_median_H10",
                                 "vm319_meas_density_H156",
                                 "vm23_time_to_last_ms",
                                 "pm83_iqr_H62"]

    execute(configs)
