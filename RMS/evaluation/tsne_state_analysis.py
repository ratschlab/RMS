''' Correlation of risk scores with the important variables
    in terms of a t-SNE embedding'''

import glob
import os
import os.path
import pickle
import ipdb
import argparse
import timeit
import random

import pandas as pd
import numpy as np
import numpy.random as np_rand
import umap
import scipy.stats as sp_stats
import sklearn.impute as sk_impute
import sklearn.decomposition as sk_decomp
import sklearn.utils as sk_utils
import sklearn.manifold as sk_manifold
import sklearn.preprocessing as sk_preproc
import sklearn.cluster as sk_cluster
import sklearn.neighbors as sk_neighbors

import matplotlib as mpl
#mpl.use("Agg")
mpl.use("Cairo")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def scale_unit_interval(x_arr,min_val=0,max_val=1):
    x_std = (x_arr - x_arr.min()) / (x_arr.max() - x_arr.min())
    x_scaled = x_std * (max_val - min_val) + min_val
    return x_scaled

def execute(configs):

    random.seed(2023)
    np_rand.seed(2023)
    
    with open(configs["split_path"],'rb') as fp:
        splits=pickle.load(fp)

    test_pids=splits[configs["split_key"]]["test"]

    with open(configs["batch_path"],'rb') as fp:
        batch_desc=pickle.load(fp)["pid_to_chunk"]

    test_chunks=list(sorted(set([batch_desc[pid] for pid in test_pids])))

    df_early_ef_pred_desc=pd.read_parquet(configs["early_ef_pred_desc"])
    early_ef_pred_pids=df_early_ef_pred_desc["pid"].values.tolist()
    early_ef_pred_timestamp=list(df_early_ef_pred_desc["timestamp"].values)
    early_ef_pred_all=list(zip(early_ef_pred_pids,early_ef_pred_timestamp))

    print("Number of test PIDs: {}".format(len(test_pids)))

    color_rf_pred_collect=[]
    color_ef_pred_collect=[]
    color_rexp_pred_collect=[]
    color_vent_pred_collect=[]

    color_rf_true_collect=[]
    color_ef_true_collect=[]
    color_rexp_true_collect=[]
    color_vent_true_collect=[]
    color_mort_true_collect=[]
    color_early_ef_pred_collect=[]
    
    input_collect=[]

    static_f=pd.read_hdf(os.path.join(configs["ml_input_path"],"reduced",
                                      configs["split_key"],"static.h5"),mode='r')
    for batch_idx in test_chunks:

        print("Processing batch {}".format(batch_idx))

        # Collect the prediction files of each task
        pred_rf_batch_f=os.path.join(configs["pred_path"],"reduced",
                                  configs["split_key"],configs["rf_model_key"],
                                  "batch_{}.h5".format(batch_idx))

        pred_ef_batch_f=os.path.join(configs["pred_path"],"reduced",
                                  configs["split_key"],configs["ef_model_key"],
                                  "batch_{}.h5".format(batch_idx))

        pred_rexp_batch_f=os.path.join(configs["pred_path"],"reduced",
                                  configs["split_key"],configs["rexp_model_key"],
                                  "batch_{}.h5".format(batch_idx))

        pred_vent_batch_f=os.path.join(configs["pred_path"],"reduced",
                                  configs["split_key"],configs["vent_model_key"],
                                  "batch_{}.h5".format(batch_idx))        
        
        input_batch_f=os.path.join(configs["ml_input_path"],"reduced",
                                   configs["split_key"],"batch_{}.h5".format(batch_idx))

        with pd.HDFStore(pred_rf_batch_f,'r') as pred_store:
            pred_ks=list(pred_store.keys())

        with pd.HDFStore(input_batch_f,'r') as pred_store:
            input_ks=list(pred_store.keys())

        important_vars=configs["IMPORTANT_VARS"]
        unique_pred_pids=list(map(lambda key: int(key[2:]), pred_ks))

        df_feat=pd.read_hdf(os.path.join(configs["ml_input_path"],"reduced",
                                         configs["split_key"],"batch_{}.h5".format(batch_idx)),
                            '/imputed',mode='r')[["PatientID","AbsDatetime","RelDatetime"]+important_vars]
        unique_input_pids=list(df_feat["PatientID"].unique())

        batch_test_pids=list(set(test_pids).intersection(set(unique_pred_pids)))
        batch_test_pids=list(set(batch_test_pids).intersection(set(unique_input_pids)))

        print("Number of test PIDs in the batch: {}".format(len(batch_test_pids)))

        for pid in batch_test_pids:
            df_feat_pid=df_feat[df_feat["PatientID"]==pid]

            # Prediction frame for patient
            df_rf_pred_pid=pd.read_hdf(pred_rf_batch_f,'/p{}'.format(pid),mode='r')[["PatientID","AbsDatetime","RelDatetime","PredScore","PredLabel","TrueLabel"]]
            df_ef_pred_pid=pd.read_hdf(pred_ef_batch_f,'/p{}'.format(pid),mode='r')[["PatientID","AbsDatetime","RelDatetime","PredScore","PredLabel","TrueLabel"]]
            df_rexp_pred_pid=pd.read_hdf(pred_rexp_batch_f,'/p{}'.format(pid),mode='r')[["PatientID","AbsDatetime","RelDatetime","PredScore","PredLabel","TrueLabel"]]
            df_vent_pred_pid=pd.read_hdf(pred_vent_batch_f,'/p{}'.format(pid),mode='r')[["PatientID","AbsDatetime","RelDatetime","PredScore","PredLabel","TrueLabel"]]            
            
            assert df_feat_pid.shape[0]==df_rf_pred_pid.shape[0]
            assert df_feat_pid.shape[0]==df_ef_pred_pid.shape[0]
            assert df_feat_pid.shape[0]==df_rexp_pred_pid.shape[0]
            assert df_feat_pid.shape[0]==df_vent_pred_pid.shape[0]

            abs_timestamps=list(df_feat_pid["AbsDatetime"].values)

            df_static_pid=static_f[static_f.PatientID==pid]
            mort_status=df_static_pid.iloc[0].Discharge
            
            df_rf_colors=df_rf_pred_pid
            df_ef_colors=df_ef_pred_pid
            df_rexp_colors=df_rexp_pred_pid
            df_vent_colors=df_vent_pred_pid            
            
            df_inputs=df_feat_pid

            color_rf_pred_vect=df_rf_colors["PredScore"].values
            color_ef_pred_vect=df_ef_colors["PredScore"].values
            color_rexp_pred_vect=df_rexp_colors["PredScore"].values
            color_vent_pred_vect=df_vent_colors["PredScore"].values

            color_rf_true_vect=df_rf_colors["TrueLabel"].values
            color_ef_true_vect=df_ef_colors["TrueLabel"].values
            color_rexp_true_vect=df_rexp_colors["TrueLabel"].values
            color_vent_true_vect=df_vent_colors["TrueLabel"].values

            color_mort_true_vect=np.zeros_like(color_vent_true_vect)
            if mort_status==4:
                color_mort_true_vect[-12*24:]=1

            # Construct a vect for early EF predictions
            if pid not in early_ef_pred_pids:
                color_early_ef_pred_true_vect=np.zeros_like(color_vent_true_vect)
            else:
                color_early_ef_pred_true_vect=np.zeros_like(color_vent_true_vect)
                for tidx in range(len(abs_timestamps)):
                    if (pid,abs_timestamps[tidx]) in early_ef_pred_all:
                        df_save_filter=df_early_ef_pred_desc[(df_early_ef_pred_desc.pid==pid) & (df_early_ef_pred_desc.timestamp==abs_timestamps[tidx])]
                        hour_offset=df_save_filter.iloc[0]["hour_offset"]
                        color_early_ef_pred_true_vect[max(0,tidx-int(hour_offset*12)):tidx]=1
            
            in_vect=df_inputs[important_vars].values

            color_rf_pred_collect.append(color_rf_pred_vect)
            color_ef_pred_collect.append(color_ef_pred_vect)
            color_rexp_pred_collect.append(color_rexp_pred_vect)
            color_vent_pred_collect.append(color_vent_pred_vect)

            color_rf_true_collect.append(color_rf_true_vect)
            color_ef_true_collect.append(color_ef_true_vect)
            color_rexp_true_collect.append(color_rexp_true_vect)
            color_vent_true_collect.append(color_vent_true_vect)
            color_mort_true_collect.append(color_mort_true_vect)
            color_early_ef_pred_collect.append(color_early_ef_pred_true_vect)
            
            input_collect.append(in_vect)

        if configs["debug_mode"]:
            break

    X_arr=np.concatenate(input_collect,axis=0)
    score_pred_rf_arr=np.concatenate(color_rf_pred_collect,axis=0)
    score_pred_ef_arr=np.concatenate(color_ef_pred_collect,axis=0)
    score_pred_rexp_arr=np.concatenate(color_rexp_pred_collect,axis=0)
    score_pred_vent_arr=np.concatenate(color_vent_pred_collect,axis=0)
    
    score_true_rf_arr=np.concatenate(color_rf_true_collect,axis=0)
    score_true_ef_arr=np.concatenate(color_ef_true_collect,axis=0)
    score_true_rexp_arr=np.concatenate(color_rexp_true_collect,axis=0)
    score_true_vent_arr=np.concatenate(color_vent_true_collect,axis=0)
    score_true_mort_arr=np.concatenate(color_mort_true_collect,axis=0)
    score_true_early_ef_pred_arr=np.concatenate(color_early_ef_pred_collect,axis=0)

    status_vent_arr=np.isnan(score_true_vent_arr).astype(float)
    status_rf_arr=np.isnan(score_true_rf_arr).astype(float)
    status_rexp_arr=(np.isnan(score_true_vent_arr) & np.isnan(score_true_rexp_arr)).astype(float)
    status_rexp_arr[np.isfinite(score_true_vent_arr)]=np.nan

    imputer=sk_impute.SimpleImputer()
    std_scaler=sk_preproc.StandardScaler()
    X_arr_orig=imputer.fit_transform(X_arr)
    X_arr=std_scaler.fit_transform(X_arr_orig)
    print("Input dimensions: {}".format(X_arr.shape))

    if configs["transform_type"]=="umap":
        reducer=umap.UMAP()
    elif configs["transform_type"]=="pca":
        reducer=sk_decomp.PCA(n_components=2)
    elif configs["transform_type"]=="tsne":
        reducer=sk_manifold.TSNE(n_components=2,random_state=2023)
    
    print("Computing embedding...")

    if configs["small_sample"]:
        X_arr,X_arr_orig,score_pred_rf_arr,score_pred_ef_arr,\
            score_pred_rexp_arr,score_pred_vent_arr,\
            score_true_rf_arr,score_true_ef_arr, \
            score_true_rexp_arr,score_true_vent_arr, \
            score_true_mort_arr, score_true_early_ef_pred_arr, status_rf_arr, \
            status_vent_arr, status_rexp_arr = sk_utils.shuffle(X_arr,X_arr_orig,score_pred_rf_arr,score_pred_ef_arr,
                                                                score_pred_rexp_arr,score_pred_vent_arr,
                                                                score_true_rf_arr,score_true_ef_arr,
                                                                score_true_rexp_arr,score_true_vent_arr,score_true_mort_arr,score_true_early_ef_pred_arr, status_rf_arr,
                                                                status_vent_arr, status_rexp_arr,n_samples=150000, # 150,000 can be used, 1000 for debugging
                                                                random_state=2023)

    t_begin=timeit.default_timer()
    X_emb=reducer.fit_transform(X_arr)
    t_end=timeit.default_timer()
    print("Embedding compute time: {:.2f} secs".format(t_end-t_begin))

    for outcome_desc,pred_score_A_arr,pred_true_A_arr, \
        in [("risk_rf",score_pred_rf_arr,score_true_rf_arr),
            ("risk_ef",score_pred_ef_arr,score_true_ef_arr),  
            ("risk_rexp",score_pred_rexp_arr,score_true_rexp_arr),
            ("risk_vent",score_pred_vent_arr,score_true_vent_arr),
            ("status_vent",status_vent_arr,status_vent_arr),
            ("status_rf",status_rf_arr,status_rf_arr),
            ("status_rexp",status_rexp_arr,status_rexp_arr),
            ("status_mort",score_true_mort_arr,score_true_mort_arr),
            ("status_early_ef_pred",score_true_early_ef_pred_arr,score_true_early_ef_pred_arr),
            ("risk_rf_vs_ef",score_pred_rf_arr,score_true_rf_arr),
            ("risk_rf_vs_rexp",score_pred_rf_arr,score_true_rf_arr),    
            ("risk_rf_vs_vent",score_pred_rf_arr,score_true_rf_arr)]:

        if outcome_desc in ["risk_rf","risk_rexp","risk_vent","status_vent","status_rexp","status_mort","status_rf"]:
            filter_arr=np.isfinite(pred_score_A_arr) & np.isfinite(pred_true_A_arr)
        elif outcome_desc in ["risk_ef","status_early_ef_pred"]:
            filter_arr=np.isfinite(pred_score_A_arr) & np.isnan(score_true_vent_arr) & np.isnan(score_true_rexp_arr)
        elif outcome_desc=="risk_rf_vs_vent":
            filter_arr=np.isfinite(pred_score_A_arr) & np.isfinite(pred_true_A_arr) & np.isfinite(score_pred_vent_arr) & np.isfinite(score_true_vent_arr)
        elif outcome_desc=="risk_rf_vs_rexp":
            filter_arr=np.isfinite(pred_score_A_arr) & np.isfinite(pred_true_A_arr) & np.isfinite(score_pred_rexp_arr) & np.isfinite(score_true_rexp_arr)
        elif outcome_desc=="risk_rf_vs_ef":
            filter_arr=np.isfinite(pred_score_A_arr) & np.isfinite(pred_true_A_arr) & np.isfinite(score_pred_ef_arr) & np.isnan(score_true_vent_arr) & np.isnan(score_true_rexp_arr)
            
        X_emb_red=X_emb[filter_arr]
        X_arr_red=X_arr_orig[filter_arr]
        score_arr_A=pred_score_A_arr[filter_arr]

        print("Outcome: {}, Number of scores: {}".format(outcome_desc,score_arr_A.size))

        # Create a number of hex plots, just to extract the median values of the 17 variables and the 4 risk scores
        filt_pred_rf_arr=score_pred_rf_arr[filter_arr]
        filt_pred_ef_arr=score_pred_ef_arr[filter_arr]
        filt_pred_rexp_arr=score_pred_rexp_arr[filter_arr]
        filt_pred_vent_arr=score_pred_vent_arr[filter_arr]
        filt_status_vent_arr=status_vent_arr[filter_arr]
        filt_status_rexp_arr=status_rexp_arr[filter_arr]
        filt_status_rf_arr=status_rf_arr[filter_arr]
        filt_status_mort_arr=score_true_mort_arr[filter_arr]
        filt_status_early_ef_pred_arr=score_true_early_ef_pred_arr[filter_arr]

        for score_name,pred_score_arr in [("status_vent",filt_status_vent_arr),
                                          ("status_rexp",filt_status_rexp_arr),
                                          ("status_rf",filt_status_rf_arr),
                                          ("rf",filt_pred_rf_arr),
                                          ("ef",filt_pred_ef_arr),
                                          ("status_mort",filt_status_mort_arr),
                                          ("status_early_ef_pred",filt_status_early_ef_pred_arr),
                                          ("rexp",filt_pred_rexp_arr),
                                          ("vent", filt_pred_vent_arr)]:

            hex_plot_score_Q1=plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], C=pred_score_arr, cmap="Spectral_r",
                                          gridsize=20,mincnt=30,reduce_C_function=lambda arr: np.percentile(arr,25)) # 30 should be used on 150k sampl 
            hex_plot_score_Q3=plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], C=pred_score_arr, cmap="Spectral_r",
                                          gridsize=20,mincnt=30,reduce_C_function=lambda arr: np.percentile(arr,75)) # 30 should be used on 150k sample           
            offset_arr_Q1=hex_plot_score_Q1.get_offsets()
            value_arr_Q1=hex_plot_score_Q1.get_array()
            offset_arr_Q3=hex_plot_score_Q3.get_offsets()
            value_arr_Q3=hex_plot_score_Q3.get_array()        
            df_desc_Q1=pd.DataFrame({"x_coord": offset_arr_Q1[:,0], "y_coord": offset_arr_Q1[:,1],
                                      "q1_value": value_arr_Q1})
            df_desc_Q3=pd.DataFrame({"x_coord": offset_arr_Q3[:,0], "y_coord": offset_arr_Q3[:,1],
                                      "q3_value": value_arr_Q3})
            df_desc_Q1.to_parquet(os.path.join(configs["plot_path"],"{}_{}_Q1.parquet".format(outcome_desc,score_name)))
            df_desc_Q3.to_parquet(os.path.join(configs["plot_path"],"{}_{}_Q3.parquet".format(outcome_desc,score_name)))            
            plt.clf()

            if score_name in ["status_vent","status_rexp","status_mort","status_rf","status_early_ef_pred"]:
                hex_plot_score_mean=plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], C=pred_score_arr, cmap="Spectral_r",
                                                 gridsize=20,mincnt=30,reduce_C_function=np.mean) # 30 should be used on 150k sample
            else:
                hex_plot_score_mean=plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], C=pred_score_arr, cmap="Spectral_r",
                                                 gridsize=20,mincnt=30,reduce_C_function=np.median) # 30 should be used on 150k sample                
                
            plt.colorbar()
            plt.savefig(os.path.join(configs["plot_path"],"{}_{}.png".format(outcome_desc,score_name)))
            offset_arr_mean=hex_plot_score_mean.get_offsets()
            value_arr_mean=hex_plot_score_mean.get_array()
            df_desc_mean=pd.DataFrame({"x_coord": offset_arr_mean[:,0], "y_coord": offset_arr_mean[:,1],
                                       "median_value": value_arr_mean})
            df_desc_mean.to_parquet(os.path.join(configs["plot_path"],"{}_{}_MEDIAN.parquet".format(outcome_desc,score_name)))
            plt.clf()

        for vidx,var_name in enumerate(important_vars):
            hex_plot_input_mean=plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], C=X_arr_red[:,vidx], cmap="Spectral_r",
                                           gridsize=20,mincnt=30,reduce_C_function=np.median) # 30 should be used on 150k sample
            offset_arr_mean=hex_plot_input_mean.get_offsets()
            value_arr_mean=hex_plot_input_mean.get_array()
            df_desc_mean=pd.DataFrame({"x_coord": offset_arr_mean[:,0], "y_coord": offset_arr_mean[:,1],
                                       "median_value": value_arr_mean})
            df_desc_mean.to_parquet(os.path.join(configs["plot_path"],"{}_{}_MEDIAN.parquet".format(outcome_desc,var_name)))
            plt.clf()
            hex_plot_input_Q1=plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], C=X_arr_red[:,vidx], cmap="Spectral_r",
                                           gridsize=20,mincnt=30,reduce_C_function=lambda arr: np.percentile(arr,25)) # 30 should be used on 150k sample 
            hex_plot_input_Q3=plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], C=X_arr_red[:,vidx], cmap="Spectral_r",
                                           gridsize=20,mincnt=30,reduce_C_function=lambda arr: np.percentile(arr,75)) # 30 should be used on 150k sample           
            offset_arr_Q1=hex_plot_input_Q1.get_offsets()
            value_arr_Q1=hex_plot_input_Q1.get_array()
            offset_arr_Q3=hex_plot_input_Q3.get_offsets()
            value_arr_Q3=hex_plot_input_Q3.get_array()            
            df_desc_Q1=pd.DataFrame({"x_coord": offset_arr_Q1[:,0], "y_coord": offset_arr_Q1[:,1],
                                      "q1_value": value_arr_Q1})
            df_desc_Q3=pd.DataFrame({"x_coord": offset_arr_Q3[:,0], "y_coord": offset_arr_Q3[:,1],
                                      "q3_value": value_arr_Q3})            
            df_desc_Q1.to_parquet(os.path.join(configs["plot_path"],"{}_{}_Q1.parquet".format(outcome_desc,var_name)))
            df_desc_Q3.to_parquet(os.path.join(configs["plot_path"],"{}_{}_Q3.parquet".format(outcome_desc,var_name)))
        
        plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], cmap='Spectral_r',gridsize=20,mincnt=1) 
        plt.colorbar()
        plt.title("Density in the hex bins",fontsize=14)
        plt.savefig(os.path.join(configs["plot_path"],"hexbin_density_{}_{}_plot.png".format(configs["transform_type"], outcome_desc)),dpi=200)
        plt.clf()

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--pred_path", default="../../data/predictions",
                        help="Path from which predictions should be loaded")

    parser.add_argument("--ml_input_path", default="../../data/imputed/impute_hirid2",
                        help="Path from which ML input should be loaded")

    parser.add_argument("--split_path", default="../../data/exp_design/temp_splits_hirid2.pickle",
                        help="Split descriptor")

    parser.add_argument("--batch_path", default="../../data/exp_design/hirid2_chunking_100.pickle")

    parser.add_argument("--early_ef_pred_desc", default="../../data/plots/earliness_ef/early_pids_timestamps.parquet",
                        help="PIDs/timestamps of early EF predictions")

    # Output paths
    parser.add_argument("--plot_path", default="../../data/plots/umap",
                        help="Path where to save plots")

    # Arguments
    parser.add_argument("--split_key", default="temporal_1", help="Split to use for data analysis")

    parser.add_argument("--rf_model_key", default="Label_WorseStateFromZeroOrOne0To24Hours_internal_rmsRF_lightgbm",
                        help="Model key to analyze")

    parser.add_argument("--ef_model_key", default="Label_ExtubationFailureSimple_internal_rmsEF_lightgbm",
                       help="Model key to analyze")

    parser.add_argument("--rexp_model_key", default="Label_ReadyExtubate0To24Hours_internal_rmsREXP_lightgbm",
                       help="Model key to analyze")

    parser.add_argument("--vent_model_key", default="Label_Ventilation0To24Hours_internal_rmsVENT_lightgbm",
                       help="Model key to analyze")     

    parser.add_argument("--debug_mode", default=False, action="store_true",
                        help="Should debug mode be used?")

    parser.add_argument("--transform_type", default="tsne", help="Transform to use")

    parser.add_argument("--small_sample", default=True, action="store_true")

    configs=vars(parser.parse_args())


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

    configs["NON_VENT_CELLS"]=[240,190]
    configs["VENT_CELLS"]=[0,1,3,16,45,66,76]

    execute(configs)
