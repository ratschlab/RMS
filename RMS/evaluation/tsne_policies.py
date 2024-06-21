''' Analyze HiRID-II/UMCDB policy differences with a t-SNE plot'''

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
    
    with open(configs["hirid_split_path"],'rb') as fp:
        hirid_splits=pickle.load(fp)

    with open(configs["umcdb_split_path"],'rb') as fp:
        umc_splits=pickle.load(fp)        

    hirid_test_pids=hirid_splits[configs["hirid_split_key"]]["test"]
    umc_test_pids=umc_splits[configs["umcdb_split_key"]]["test"]

    print("Number of HIRID/UMCDB patients: {}/{}".format(len(hirid_test_pids), len(umc_test_pids)))

    with open(configs["hirid_batch_path"],'rb') as fp:
        hirid_batch_desc=pickle.load(fp)["pid_to_chunk"]

    with open(configs["umcdb_batch_path"],'rb') as fp:
        umc_batch_desc=pickle.load(fp)["pid_to_chunk"]        

    hirid_test_chunks=list(sorted(set([hirid_batch_desc[pid] for pid in hirid_test_pids])))
    umc_test_chunks=list(sorted(set([umc_batch_desc[pid] for pid in umc_test_pids])))    

    hirid_input_collect=[]
    umc_input_collect=[]
    hirid_drug_collect=[]
    umc_drug_collect=[]

    hirid_rf_true_collect=[]
    umc_rf_true_collect=[]
    hirid_rext_true_collect=[]
    umc_rext_true_collect=[]
    hirid_vent_true_collect=[]
    umc_vent_true_collect=[]

    important_vars=configs["IMPORTANT_VARS"]
    drug_vars=configs["DRUGS"]    

    # Collect HIRID-II data
    for batch_idx in hirid_test_chunks:
        print("Processing HiRID batch {}".format(batch_idx))
        input_batch_f=os.path.join(configs["hirid_path"],"reduced",
                                   configs["hirid_split_key"],"batch_{}.h5".format(batch_idx))
        pred_rf_batch_f=os.path.join(configs["pred_path"],"reduced",
                                  configs["hirid_split_key"],configs["hirid_RF_model_key"],
                                  "batch_{}.h5".format(batch_idx))
        pred_rext_batch_f=os.path.join(configs["pred_path"],"reduced",
                                  configs["hirid_split_key"],configs["hirid_REXT_model_key"],
                                  "batch_{}.h5".format(batch_idx))
        pred_vent_batch_f=os.path.join(configs["pred_path"],"reduced",
                                       configs["hirid_split_key"],configs["hirid_VENT_model_key"],
                                       "batch_{}.h5".format(batch_idx))                

        df_feat=pd.read_hdf(os.path.join(configs["hirid_path"],"reduced",
                                         configs["hirid_split_key"],"batch_{}.h5".format(batch_idx)),
                            '/imputed',mode='r')[["PatientID","AbsDatetime","RelDatetime"]+important_vars+drug_vars]
        unique_input_pids=list(df_feat["PatientID"].unique())
        
        batch_test_pids=list(set(hirid_test_pids).intersection(set(unique_input_pids)))
        print("Number of test PIDs in the batch: {}".format(len(batch_test_pids)))

        for pid in batch_test_pids:
            df_feat_pid=df_feat[df_feat["PatientID"]==pid]
            df_rf_pred_pid=pd.read_hdf(pred_rf_batch_f,'/p{}'.format(pid),mode='r')[["PatientID","AbsDatetime","RelDatetime","PredScore","PredLabel","TrueLabel"]]
            df_rext_pred_pid=pd.read_hdf(pred_rext_batch_f,'/p{}'.format(pid),mode='r')[["PatientID","AbsDatetime","RelDatetime","PredScore","PredLabel","TrueLabel"]]
            df_vent_pred_pid=pd.read_hdf(pred_vent_batch_f,'/p{}'.format(pid),mode='r')[["PatientID","AbsDatetime","RelDatetime","PredScore","PredLabel","TrueLabel"]]                        
            assert df_feat_pid.shape[0]==df_rf_pred_pid.shape[0]
            assert df_feat_pid.shape[0]==df_rext_pred_pid.shape[0]
            assert df_feat_pid.shape[0]==df_vent_pred_pid.shape[0]                             
            df_inputs=df_feat_pid
            
            df_rf_colors=df_rf_pred_pid
            color_rf_true_vect=df_rf_colors["TrueLabel"].values
            df_rext_colors=df_rext_pred_pid
            color_rext_true_vect=df_rext_colors["TrueLabel"].values
            df_vent_colors=df_vent_pred_pid
            color_vent_true_vect=df_vent_colors["TrueLabel"].values
            
            in_vect=df_inputs[important_vars].values
            drug_vect=df_inputs[drug_vars].values
            hirid_input_collect.append(in_vect)
            hirid_drug_collect.append(drug_vect)
            hirid_rf_true_collect.append(color_rf_true_vect)
            hirid_rext_true_collect.append(color_rext_true_vect)
            hirid_vent_true_collect.append(color_vent_true_vect)

        if configs["debug_mode"]:
            break

    # Collect UMCDB data
    for batch_idx in umc_test_chunks:
        print("Processing UMCDB batch {}".format(batch_idx))
        input_batch_f=os.path.join(configs["umcdb_path"],"reduced",
                                   configs["umcdb_split_key"],"batch_{}.h5".format(batch_idx))
        pred_rf_batch_f=os.path.join(configs["pred_path"],"reduced",
                                  configs["hirid_split_key"],configs["umcdb_RF_model_key"],
                                  "batch_{}.h5".format(batch_idx))
        pred_rext_batch_f=os.path.join(configs["pred_path"],"reduced",
                                       configs["hirid_split_key"],configs["umcdb_REXT_model_key"],
                                       "batch_{}.h5".format(batch_idx))
        pred_vent_batch_f=os.path.join(configs["pred_path"],"reduced",
                                       configs["hirid_split_key"],configs["umcdb_VENT_model_key"],
                                       "batch_{}.h5".format(batch_idx))                

        if not os.path.exists(pred_rf_batch_f):
            print("Skipping RF batch {}".format(batch_idx))
            continue

        if not os.path.exists(pred_rext_batch_f):
            print("Skipping REXT batch {}".format(batch_idx))
            continue

        if not os.path.exists(pred_vent_batch_f):
            print("Skipping REXT batch {}".format(batch_idx))
            continue                
        
        df_feat=pd.read_hdf(os.path.join(configs["umcdb_path"],"reduced",
                                         configs["umcdb_split_key"],"batch_{}.h5".format(batch_idx)),
                            '/imputed',mode='r')[["PatientID","AbsDatetime","RelDatetime"]+important_vars+drug_vars]
        unique_input_pids=list(df_feat["PatientID"].unique())
        batch_test_pids=list(set(umc_test_pids).intersection(set(unique_input_pids)))
        print("Number of test PIDs in the batch: {}".format(len(batch_test_pids)))

        for pid in batch_test_pids:
            df_feat_pid=df_feat[df_feat["PatientID"]==pid]
            try:
                df_rf_pred_pid=pd.read_hdf(pred_rf_batch_f,'/p{}'.format(pid),mode='r')[["PatientID","AbsDatetime","RelDatetime","PredScore","PredLabel","TrueLabel"]]
                rf_present=True
                df_rf_colors=df_rf_pred_pid
                assert df_feat_pid.shape[0]==df_rf_pred_pid.shape[0]                
            except KeyError:
                rf_present=False

            try:
                df_rext_pred_pid=pd.read_hdf(pred_rext_batch_f,'/p{}'.format(pid),mode='r')[["PatientID","AbsDatetime","RelDatetime","PredScore","PredLabel","TrueLabel"]]
                rext_present=True
                df_rext_colors=df_rext_pred_pid
                assert df_feat_pid.shape[0]==df_rext_pred_pid.shape[0]                            
            except KeyError:
                rext_present=False

            try:
                df_vent_pred_pid=pd.read_hdf(pred_vent_batch_f,'/p{}'.format(pid),mode='r')[["PatientID","AbsDatetime","RelDatetime","PredScore","PredLabel","TrueLabel"]]
                vent_present=True
                df_vent_colors=df_vent_pred_pid
                assert df_feat_pid.shape[0]==df_vent_pred_pid.shape[0]                            
            except KeyError:
                vent_present=False

            df_inputs=df_feat_pid

            if rf_present:
                color_rf_true_vect=df_rf_colors["TrueLabel"].values
            else:
                color_rf_true_vect=np.zeros(df_feat_pid.shape[0])
                color_rf_true_vect[:]=np.nan

            if rext_present:
                color_rext_true_vect=df_rext_colors["TrueLabel"].values
            else:
                color_rext_true_vect=np.zeros(df_feat_pid.shape[0])
                color_rext_true_vect[:]=np.nan

            if vent_present:
                color_vent_true_vect=df_vent_colors["TrueLabel"].values
            else:
                color_vent_true_vect=np.zeros(df_feat_pid.shape[0])
                color_vent_true_vect[:]=np.nan                
            
            in_vect=df_inputs[important_vars].values 
            drug_vect=df_inputs[drug_vars].values           
            umc_input_collect.append(in_vect)
            umc_drug_collect.append(drug_vect)
            umc_rf_true_collect.append(color_rf_true_vect)
            umc_rext_true_collect.append(color_rext_true_vect)
            umc_vent_true_collect.append(color_vent_true_vect)

        if configs["debug_mode"]:
            break

    hirid_X_arr=np.concatenate(hirid_input_collect,axis=0)
    umcdb_X_arr=np.concatenate(umc_input_collect,axis=0)

    hirid_drug_arr=np.concatenate(hirid_drug_collect,axis=0)
    umcdb_drug_arr=np.concatenate(umc_drug_collect,axis=0)

    hirid_RF_arr=np.concatenate(hirid_rf_true_collect,axis=0)
    umcdb_RF_arr=np.concatenate(umc_rf_true_collect,axis=0)

    hirid_REXT_arr=np.concatenate(hirid_rext_true_collect,axis=0)
    umcdb_REXT_arr=np.concatenate(umc_rext_true_collect,axis=0)

    hirid_VENT_arr=np.concatenate(hirid_vent_true_collect,axis=0)
    umcdb_VENT_arr=np.concatenate(umc_vent_true_collect,axis=0)            

    n_umc_samples=umcdb_X_arr.shape[0]
    hirid_X_arr,hirid_drug_arr,hirid_RF_arr,hirid_REXT_arr,hirid_VENT_arr=sk_utils.shuffle(hirid_X_arr,hirid_drug_arr,hirid_RF_arr,hirid_REXT_arr,hirid_VENT_arr,n_samples=n_umc_samples,random_state=2023)
    
    X_arr=np.concatenate([hirid_X_arr,umcdb_X_arr],axis=0)
    drug_arr=np.concatenate([hirid_drug_arr,umcdb_drug_arr],axis=0)
    dset_arr=np.array([0]*hirid_X_arr.shape[0]+[1]*umcdb_X_arr.shape[0])
    RF_arr=np.concatenate([hirid_RF_arr,umcdb_RF_arr],axis=0)
    REXT_arr=np.concatenate([hirid_REXT_arr,umcdb_REXT_arr],axis=0)
    VENT_arr=np.concatenate([hirid_VENT_arr,umcdb_VENT_arr],axis=0)        

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
        X_arr,X_arr_orig,drug_arr,dset_arr,RF_arr,REXT_arr,VENT_arr = sk_utils.shuffle(X_arr,X_arr_orig,drug_arr,dset_arr,RF_arr,REXT_arr,VENT_arr,
                                                                                       n_samples=150000, # 150,000 can be used, 1000 for debugging
                                                                                       random_state=2023)

    t_begin=timeit.default_timer()
    X_emb=reducer.fit_transform(X_arr)
    t_end=timeit.default_timer()
    print("Embedding compute time: {:.2f} secs".format(t_end-t_begin))

    color_map_drug=dict()    

    for outcome_desc,filter_desc in [("all_samples","RF"),("all_samples","EF"),
                                     ("hirid_samples","RF"),("hirid_samples","EF"),
                                     ("umcdb_samples","RF"),("umcdb_samples","EF")]:

        if filter_desc=="RF":
            filter_ep=np.isfinite(RF_arr)
        elif filter_desc=="EF":
            filter_ep=np.isnan(REXT_arr) & np.isnan(VENT_arr)
        else:
            assert False
        
        if not outcome_desc=="all_samples":
            if outcome_desc=="hirid_samples":
                filter_arr=(dset_arr==0) & filter_ep
            elif outcome_desc=="umcdb_samples":
                filter_arr=(dset_arr==1) & filter_ep
            X_emb_red=X_emb[filter_arr]
            X_arr_red=X_arr_orig[filter_arr]
            X_drug_red=drug_arr[filter_arr]
        else:
            filter_arr=filter_ep
            X_emb_red=X_emb[filter_arr]
            X_arr_red=X_arr_orig[filter_arr]
            X_drug_red=drug_arr[filter_arr]
            X_dset_red=dset_arr[filter_arr]

        for vidx,var_name in enumerate(drug_vars):
            if outcome_desc=="all_samples":
                hex_plot_input_mean=plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], C=X_drug_red[:,vidx], cmap="Spectral_r",
                                               gridsize=20,mincnt=20,reduce_C_function=np.mean) # 20 should be used on 150k sample
                color_map_drug[var_name]=(hex_plot_input_mean.get_array().min(), hex_plot_input_mean.get_array().max())
                plt.clf()
            else:
                hex_plot_input_mean=plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], C=X_drug_red[:,vidx], cmap="Spectral_r",
                                               vmin=color_map_drug[var_name][0],vmax=color_map_drug[var_name][1],
                                               gridsize=20,mincnt=20,reduce_C_function=np.mean) # 20 should be used on 150k sample
                plt.colorbar()
                plt.savefig(os.path.join(configs["plot_path"],"var_mean_{}_{}_{}_{}_plot.png".format(configs["transform_type"], filter_desc,outcome_desc,var_name)),dpi=200)            
                plt.clf()

        if outcome_desc=="all_samples":
            hex_plot_dset_mean=plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], C=X_dset_red, cmap="Spectral_r",
                                           gridsize=20,mincnt=20,reduce_C_function=np.mean) # 20 should be used on 150k sample
            plt.colorbar()
            plt.savefig(os.path.join(configs["plot_path"],"dset_mean_{}_{}_{}_plot.png".format(configs["transform_type"],filter_desc,outcome_desc)),dpi=200)
            plt.clf()            
        
        plt.hexbin(X_emb_red[:, 0], X_emb_red[:, 1], cmap='Spectral_r',gridsize=20,mincnt=1)
        plt.colorbar()
        plt.title("Density in the hex bins",fontsize=14)
        plt.savefig(os.path.join(configs["plot_path"],"hexbin_density_{}_{}_{}_plot.png".format(configs["transform_type"], filter_desc,outcome_desc)),dpi=200)
        plt.clf()

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--hirid_path", default="../../data/imputed/impute_hirid2",
                        help="Path from which ML input should be loaded")

    parser.add_argument("--umcdb_path", default="../../data/imputed/impute_umcdb_OLD",
                        help="Path from which ML input should be loaded")

    parser.add_argument("--pred_path", default="../../data/predictions",
                        help="Path from which predictions should be loaded")



    parser.add_argument("--hirid_split_path", default="../../data/exp_design/temp_splits_hirid2.pickle",
                        help="Split descriptor")
    parser.add_argument("--umcdb_split_path", default="../../data/exp_design/random_splits_umcdb.pickle",
                        help="Split descriptor")    

    parser.add_argument("--hirid_batch_path", default="../../data/exp_design/hirid2_chunking_100.pickle")
    parser.add_argument("--umcdb_batch_path", default="../../data/exp_design/umcdb_chunking.pickle")    

    # Output paths
    parser.add_argument("--plot_path", default="../../data/plots/tsne_policy",
                        help="Path where to save plots")

    # Arguments
    parser.add_argument("--hirid_split_key", default="temporal_1", help="Split to use for data analysis")
    parser.add_argument("--umcdb_split_key", default="random_1", help="Split to use for data analysis")

    parser.add_argument("--hirid_RF_model_key", default="Label_WorseStateFromZeroOrOne0To24Hours_internal_rmsRF_lightgbm",
                        help="Model key to analyze")
    parser.add_argument("--umcdb_RF_model_key", default="Label_WorseStateFromZeroOrOne0To24Hours_val_rmsRF_lightgbm",
                        help="Model key to analyze")
    parser.add_argument("--hirid_REXT_model_key", default="Label_ReadyExtubate0To24Hours_internal_rmsREXP_lightgbm",
                        help="Model key to analyze")
    parser.add_argument("--umcdb_REXT_model_key", default="Label_ReadyExtubate0To24Hours_val_compact_lightgbm",
                        help="Model key to analyze")
    parser.add_argument("--hirid_VENT_model_key", default="Label_ReadyExtubate0To24Hours_internal_rmsREXP_lightgbm",
                        help="Model key to analyze")
    parser.add_argument("--umcdb_VENT_model_key", default="Label_Ventilation0To24Hours_val_compact_lightgbm",
                        help="Model key to analyze")        

    parser.add_argument("--debug_mode", default=False, action="store_true",
                        help="Should debug mode be used?")

    parser.add_argument("--transform_type", default="tsne", help="Transform to use")

    parser.add_argument("--small_sample", default=True, action="store_true")

    configs=vars(parser.parse_args())

    configs["DRUGS"]=["pm77", # Benzodiacepine (INCLUDED)
                      "pm80", # Propofol (INCLUDED)
                      "pm95", # Heparin (INCLUDED)
                      "pm41", # Dobutamine (INCLUDED) 
                      "pm69", # Loop diuretics (INCLUDED)                      
                      "pm83", # Insulin kurzwirksam (INCLUDED)                    
                      "pm39"] # Norepinephrine (INCLUDED)
    
    #configs["IMPORTANT_VARS"]=["vm211", # Pressure support (INCLUDED)
    #                            "vm58", # FiO2 (INCLUDED)
    #                            "vm216", # MV spont servo (INCLUDED)
    #                            "vm318", # Atemfrequenz (INCLUDED)
    #                            "vm319", # RRsp m (INCLUDED)
    #                            "vm23", # Supplemental oxygen (INCLUDED)
    #                            "vm26", # GCS Motor (INCLUDED)
    #                            "vm20", # SpO2 (INCLUDED)
    #                            "vm140", # PaO2 (INCLUDED)
    #                            "vm309", # Supplemental FiO2 % (INCLUDED)
    #                            "vm62", # Spitzendruck (INCLUDED)
    #                            "vm293"] # PEEPs (INCLUDED)

    # Only physiological parameters
    configs["IMPORTANT_VARS"]=["vm1", # Heart rate
                               "vm2", # Temperature
                               "vm5", # Mean arterial pressure
                               "vm20", # SpO2
                               "vm22", # RR
                               "vm136", # Lactate
                               "vm140"] # PaO2

    configs["NON_VENT_CELLS"]=[240,190]
    configs["VENT_CELLS"]=[0,1,3,16,45,66,76]

    execute(configs)
