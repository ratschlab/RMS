''' An analysis of signed SHAP values in the test set of a split
    printing an overview figure of the most important SHAP values'''

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
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import seaborn as sns
sns.set(style="whitegrid",font_scale=2.5)
#sns.set_context("poster")

mpl.rcParams["pdf.fonttype"]=42

def feature_to_text(feat, backup_dict=None, name_dict=None):
    ''' Converts a feature to human readable label of it'''

    resp_translate_table={"plain_fio2estimated": "FiO2 \n (Current estimate)",
                          'plain_vm309': "Suppl. \n FiO2 % (Current)",
                          'vm309_instable_l1_entire': "Supp FiO2 % (Fraction of \n previous stay in [21,40])",
                          "vm58_median_H156": "FiO2 (Median \n longest term horizon)",                          
                          'vm20_instable_l1_8h': "SpO2 (Fraction of \n last 8h [90,94] %)",
                          'plain_vm20': "SpO2 (Current)",
                          'vm309_median_H25': "Suppl. FiO2 % \n (Median mid-term horizon)",
                          "plain_vm309": "Suppl. FiO2 % (Current)",
                          "vm62_time_to_last_ms": "Vent. peak pressure \n (Time to last obs.)",
                          'plain_vm140': "PaO2 (Current)",
                          'plain_vm141': "SaO2 (Current)",
                          'Sex': "Patient gender",
                          "Age": "Patient age",
                          "vm293_median_H156": "PEEP(s) (Median \n longest-term horizon)",
                          "vm211_time_to_last_ms": "Pressure support \n (Time to last obs.)",
                          "vm26_time_to_last_ms": "GCS Motor (Time \n to last obs.)",
                          "vm318_median_H10": "Vent. RR (Median \n short-term horizon)",       
                          'vm309_median_H62': "Suppl. FiO2 % \n (Median long-term horizon)",
                          'plain_pm95': "Heparin (Current dose)",
                          'vm20_instable_l1_entire': "SpO2 (Fraction of previous \n stay in [90,94] %)",
                          'pm69_median_H62': "(Loop diuretics (Median \n dose long-term horizon)",
                          'vm23_median_H25': "Suppl. Oxygen (Median \n dose mid-term horizon)",
                          'vm23_median_H62': "Suppl. Oxygen (Median \n long-term horizon)",
                          "pm80_iqr_H62": "Propofol (IQR \n dose long-term horizon)"}

    ext_translate_table={'pm77_iqr_H10': "Benzodiacepine (IQR \n dose short-term horizon)",
                         'pm39_iqr_H156': "Norepinephrine (IQR \n dose longest-term horizon)",
                         'Emergency': "Emergency admission",
                         'pm77_iqr_H25':"(Benzodiacepine (IQR \n dose mid-term horizon)",
                         "plain_vm25": "GCS Response \n (Current)",
                         'vm23_time_to_last_ms': "Suppl. oxygen \n (Time to last obs.)",                         
                         "plain_ventstate": "Ventilation status (Current)",                        
                         "vm216_median_H10": "(MV spont servo \n (Median shorm-term horizon)",
                         "vm318_median_H10": "Vent. RR (Median \n short-term horizon)", 
                         "vm22_median_H62": "RR (Median \n long-term horizon)",
                         "vm58_iqr_H156": "FiO2 (IQR \n longest-term horizon)", 
                         "vm5_time_to_last_ms": "MAP (Time \n to last obs.)",
                         "pm80_time_to_last_ms": "Propofol (Time to \n last dose)",   
                         "vm211_time_to_last_ms": "Pressure support \n (Time to last obs.)",
                         'pm77_iqr_H62': "Benzodiacepine (IQR \n dose long-term horizon)",
                         'pm77_iqr_H156': "Benzodiacepine (IQR \n dose longest-term horizon)",
                         'pm39_time_to_last_ms': "Norepinephrine \n (Time to last dose)",
                         'pm39_iqr_H10': "Norepinephrine (IQR \n dose short-term horizon)",
                         'pm69_iqr_H25': "Loop diuretics (IQR \n dose mid-term horizon)",
                         'vm319_meas_density_H156': "RR sp. m (Observation \n freq. longest-term horizon)",
                         'vm216_median_H10': "MV spont servo \n (Median short-term horizon)",
                         'pm69_iqr_H10': "Loop diuretics (IQR \n dose short-term horizon)",
                         'pm83_iqr_H62': "Insulin (fast) (IQR \n dose long-term horizon)",
                         "pm77_time_to_last_ms": "Benzodiacepine (Time \n to last dose)",
                         'pm39_median_H156': "Norepinephrine (Median \n dose longest-term horizon)"}

    resp_translate_table.update(ext_translate_table)

    return resp_translate_table[feat]

def execute(configs):
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

    df_static=pd.read_hdf(configs["static_path"], mode='r')        
    static_cols=df_static.columns.values.tolist()

    with open(configs["split_path"],'rb') as fp:
        splits=pickle.load(fp)

    if configs["task"]=="resp_fail":
        feats_to_analyze=configs["RF_FEAT_ORDER"]
    elif configs["task"]=="ext_fail":
        feats_to_analyze=configs["EF_FEAT_ORDER"]

    acc_dict={}

    for split in configs["SPLITS"]:
        print("Analyzing split: {}".format(split))
        best_feat_file=os.path.join(configs["pred_path"],split,configs["model_config"],"best_model_shapley_values.tsv")
        best_feats=pd.read_csv(best_feat_file,sep='\t',header=0)
        best_feats.sort_values(by="importance_score",ascending=False,inplace=True)
        ipdb.set_trace()
        
        kept_static_cols=list(filter(lambda col: col in feats_to_analyze, static_cols))
        kept_pred_cols=["AbsDatetime","PredScore","TrueLabel"]+list(map(lambda col: "RawShap_"+col, feats_to_analyze))
        kept_feat_cols=["AbsDatetime"]+list(filter(lambda col: col not in kept_static_cols, feats_to_analyze))
        
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
                df_merged=df_merged[df_merged["PredScore"].notnull() & df_merged["TrueLabel"].notnull()]
                
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

    final_dataframe=pd.DataFrame(acc_dict)

    # Building the plot data frame
    df_plot = pd.DataFrame()
    feature_order = []
    feature_text = []

    for column_name in feats_to_analyze:
        column_value = column_name+"_val"
        column_shap = column_name+"_SHAP"
        df_plot_var = pd.DataFrame()
        df_plot_var['y'] = final_dataframe[column_shap]
        df_plot_var['x'] = column_name
        df_plot_var['value'] = final_dataframe[column_value]
        df_plot_var = df_plot_var.loc[~df_plot_var['value'].isna()]
        df_plot_var = df_plot_var.loc[~df_plot_var['y'].isna()]        
        feature_order.append(column_name)
        feature_text.append(feature_to_text(column_name, backup_dict=backup_dict, name_dict=name_dict))
        bins = []
        bin_count = 10
        for j in range(0,bin_count):
            offset = 0
            step = (100 - 2 * offset) / bin_count
            percentile = offset + step * j
            percentile_value = np.percentile(df_plot_var['value'], percentile)
            
            #linear interpolation for a special case (plotting issue)
            if (percentile_value == 0):
                    percentile_value = 0.000001 * j
                    
            bins.append(float(percentile_value))
            j = j + 1

        df_plot_var['hue'] = np.digitize(df_plot_var['value'], bins)
        df_plot = df_plot.append(df_plot_var)

    fig, ax = plt.subplots(figsize=(19.7, 14.7))
    ax.axhline(y=0, color='k', alpha=0.7)

    print("Number of SHAP values before downsampling: {}".format(df_plot.shape[0]))

    if configs["task"]=="resp_fail":
        df_plot_scatter = df_plot.sample(frac=0.005, replace=False)
    elif configs["task"]=="ext_fail":
        df_plot_scatter = df_plot.sample(frac=0.5, replace=False)
        
    print("Number of SHAP values after downsampling: {}".format(df_plot_scatter.shape[0]))

    sns.violinplot(x="y", y="x", kind="violin", inner=None, data=df_plot, legend=False, order=feature_order, color='k', ax=ax,
                   orient='h',cut=0,scale="width")

    # Manually set alpha for the violin plot
    #for art in ax.get_children():
    #    if isinstance(art, PolyCollection):
    #        art.set_alpha(0.75)

    # Set alpha for violin plot
    plt.setp(ax.collections, alpha=0.10)
    
    sns.swarmplot(x="y", y="x", size=1.0, data=df_plot_scatter,
                  hue='hue', ax=ax, palette=sns.color_palette("viridis", 11), order=feature_order,orient='h')
    
    ax.yaxis.label.set_visible(False)
    ax.set_xlabel("SHAP value")
    ax.get_legend().remove()
    ax.set_yticklabels(feature_text)
    ax.yaxis.set_tick_params(labelsize='small')
    fig.subplots_adjust(left=0.25)
    ax.set_xlim((-0.6, 0.75))

    # Construct a color bar
    cmap = mpl.cm.get_cmap('viridis')
    cbaxes = fig.add_axes([0.813, 0.13, 0.04, 0.2])
    cbaxes.axis('off')
    cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                          orientation='vertical')
    cbaxes.text(0.5, -0.05, 'low value', fontsize=18, horizontalalignment='center', verticalalignment='center')
    cbaxes.text(0.5, 1.05, 'high value', fontsize=18, horizontalalignment='center', verticalalignment='center')
    cb.outline.set_visible(False)
    
    #plt.tight_layout()    
    fig.savefig(os.path.join(configs["plot_path"],'compact_shap_overview_{}.pdf'.format(configs["task"])), format="pdf",dpi=300)
    fig.savefig(os.path.join(configs["plot_path"],'compact_shap_overview_{}.png'.format(configs["task"])), dpi=300)    

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
    parser.add_argument("--plot_path", default="../../data/plots/oxygenation_failure/introspection",
                         help="Plotting folder")
    #parser.add_argument("--plot_path", default="../../data/plots/extubation_failure/introspection",
    #                     help="Plotting folder")    

    # Arguments
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Process one batch")
    parser.add_argument("--small_sample", default=False, action="store_true", help="Small sample?")
    
    parser.add_argument("--task", default="resp_fail", help="Task descriptor")
    #parser.add_argument("--task", default="ext_fail", help="Task descriptor")    
    
    parser.add_argument("--model_config", default="Label_WorseStateFromZeroOrOne0To24Hours_internal_rmsRF_lightgbm", help="Model to analyze")
    #parser.add_argument("--model_config", default="Label_ExtubationFailureSimple_internal_rmsEF_lightgbm", help="Model to analyze")    

    configs=vars(parser.parse_args())

    configs["SPLITS"]=["temporal_1"]

    configs["RF_FEAT_ORDER"]=["vm58_median_H156",
                              "vm20_instable_l1_8h",
                              "plain_vm140",
                              "plain_vm309",
                              "vm62_time_to_last_ms",
                              "vm23_median_H25",
                              "vm293_median_H156",
                              "vm211_time_to_last_ms",
                              "vm26_time_to_last_ms",
                              "vm318_median_H10"]

    configs["EF_FEAT_ORDER"]=["vm211_time_to_last_ms",
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
