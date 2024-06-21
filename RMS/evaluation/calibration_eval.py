''' A script to print and save calibration metrics to the file-system'''

import argparse
import os
import os.path
import ipdb
import random
import pickle
import csv

import numpy as np
import pandas as pd
import numpy.random as np_rand
import sklearn.calibration as skcal
import sklearn.metrics as skmetrics
import sklearn.linear_model as sklm
import sklearn.isotonic as skiso
import lightgbm
import scipy
import h5py

import matplotlib as mpl
mpl.use("PDF")
import matplotlib.pyplot as plt

import RMS.utils.io as mlhc_io


def rescaled_calibration_curve(y_true,y_prob, correct_factor=None, n_bins=20):
    ''' Rescaling for prevalence version of the calibration curve'''
    bin_locs=np.arange(y_prob.min(),y_prob.max(),0.05)
    act_risks=[]
    act_locs=[]

    for thr in range(1,len(bin_locs)):
        risk_lab=y_true[(y_prob>=bin_locs[thr-1]) & (y_prob<bin_locs[thr])]
        if risk_lab.size==0:
            continue
        tps=np.sum(risk_lab==1.0)
        fps=correct_factor*np.sum(risk_lab==0.0)
        act_risk=tps/(tps+fps)
        act_risks.append(act_risk)
        act_locs.append((bin_locs[thr-1]+bin_locs[thr])/2.)

    return (np.array(act_risks),np.array(act_locs))


def calibration_metrics(configs):
    random.seed(configs["random_state"])
    np_rand.seed(configs["random_state"])                
    held_out=configs["val_type"]
    cal_set=configs["calibration_set"]
    dim_reduced_str=configs["data_mode"]
    task_key=configs["task_key"]
    assert(dim_reduced_str in ["reduced","non_reduced"])
    threshold_dict={}

    if dim_reduced_str=="reduced":
        dim_reduced_data=True
    else:
        dim_reduced_data=False    

    bern_batch_map=mlhc_io.load_pickle(configs["pid_map_path"])["pid_to_chunk"]
    
    n_skipped_patients=0
    scores_dict={}
    labels_dict={}

    cal_scores_dict={}
    cal_labels_dict={}

    for ml_model, col_desc,data_split_key,split_key in configs["ALL_WORK"]:
        print("Analyzing model ({},{},{})".format(ml_model,col_desc, split_key))

        data_split=mlhc_io.load_pickle(configs["temporal_split_path"])[split_key]
        pred_pids=data_split[held_out]
        calibration_pids=data_split[cal_set]

        if configs["small_sample"]:
            pred_pids=pred_pids[:100]
            calibration_pids=calibration_pids[:100]
        
        print("Number of test PIDs: {}".format(len(pred_pids)))
        print("Number of calibration PIDs: {}".format(len(calibration_pids)))

        output_dir=os.path.join(configs["predictions_dir"],"reduced",data_split_key,"{}_{}_{}".format(task_key, col_desc, "lightgbm"))
        #output_dir=os.path.join(configs["predictions_dir"],"reduced",data_split_key,"{}_{}_{}".format(task_key, col_desc, "tree"))        
        ep_output_dir=os.path.join(configs["endpoint_path"],data_split_key)
        imputed_output_dir=os.path.join(configs["imputed_path"],"reduced",data_split_key)

        n_skipped_patients=0
        cum_pred_scores=[]
        cum_labels=[]        
        
        # Gather original circEWS prediction scores
        for pidx,pid in enumerate(pred_pids):
            
            if (pidx+1)%500==0 and configs["verbose"]:
                print("Test PID {}/{}".format(pidx+1,len(pred_pids)))
                
            if pidx>=100 and configs["debug_mode"]:
                break
            
            batch_pat=bern_batch_map[pid]

            try:
                df_pred=pd.read_hdf(os.path.join(output_dir,"batch_{}.h5".format(batch_pat)), "/p{}".format(pid), mode='r')
                valid_pred_filter=pd.notna(df_pred["TrueLabel"]) & pd.notna(df_pred["PredScore"])                
                df_pred=df_pred[valid_pred_filter]
            except (KeyError, FileNotFoundError):
                n_skipped_patients+=1
                continue            

            if ml_model in ["lightgbm","tree"]:
                pred_scores=np.array(df_pred["PredScore"])
                true_labels=np.array(df_pred["TrueLabel"])

            elif ml_model=="threshold_baseline":
                df_ep=pd.read_hdf(os.path.join(ep_output_dir,"batch_{}.h5".format(batch_pat)),mode='r',
                                  where="PatientID={}".format(pid))
                df_ep=df_ep[valid_pred_filter]
                raw_scores=df_ep["ext_ready_violation_score"].values
                pred_scores=raw_scores/35.
                true_labels=np.array(df_pred["TrueLabel"])

            elif ml_model=="sf_ratio_baseline":
                special_pred_dir="/cluster/work/grlab/clinical/hirid2/research/RESP_RELEASE/predictions/reduced/temporal_1/Label_WorseStateFromZeroOrOne0To24Hours_one_minus_spo2_fio2_ratio"
                df_sf_pred=pd.read_hdf(os.path.join(special_pred_dir,"batch_{}.h5".format(batch_pat)),"/data",mode='r')
                df_sf_pred=df_sf_pred[df_sf_pred["PatientID"]==pid]
                df_sf_pred.reset_index(inplace=True)
                df_sf_pred=df_sf_pred[valid_pred_filter]
                    
                assert df_sf_pred.shape[0]==df_pred.shape[0]
                pred_scores=df_sf_pred["Prediction"].values
                true_labels=np.array(df_pred["TrueLabel"])                

            cum_pred_scores.append(pred_scores)
            cum_labels.append(true_labels)

        all_pred_scores=np.concatenate(cum_pred_scores)

        # For decision tree
        all_pred_scores=(all_pred_scores-all_pred_scores.min())/(all_pred_scores.max()-all_pred_scores.min())
            
        scores_dict[(ml_model,col_desc,data_split_key,split_key)]=all_pred_scores
        labels_dict[(ml_model,col_desc,data_split_key,split_key)]=np.concatenate(cum_labels)

        print("Number of skipped test patients: {}".format(n_skipped_patients))
        
        cum_cal_scores=[]
        cum_cal_labels=[]
        n_skipped_patients=0

        # Fetch prediction scores on the validation set
        for pidx,pid in enumerate(calibration_pids):
            
            if (pidx+1)%500==0 and configs["verbose"]:
                print("Validation PID {}/{}".format(pidx+1,len(calibration_pids)))
                
            if pidx>=100 and configs["debug_mode"]:
                break
            
            batch_pat=bern_batch_map[pid]

            try:
                df_pred=pd.read_hdf(os.path.join(output_dir,"batch_{}.h5".format(batch_pat)), "/p{}".format(pid), mode='r')
                df_pred=df_pred[pd.notna(df_pred["TrueLabel"]) & pd.notna(df_pred["PredScore"])]
            except (KeyError, FileNotFoundError):
                n_skipped_patients+=1
                continue

            pred_scores=np.array(df_pred["PredScore"])
            true_labels=np.array(df_pred["TrueLabel"])
            cum_cal_scores.append(pred_scores)
            cum_cal_labels.append(true_labels)

        all_cal_scores=np.concatenate(cum_cal_scores)
            
        cal_scores_dict[(ml_model,col_desc,data_split_key,split_key)]=all_cal_scores
        cal_labels_dict[(ml_model,col_desc,data_split_key,split_key)]=np.concatenate(cum_cal_labels)

        print("Number of skipped val patients: {}".format(n_skipped_patients))
        

    MODELS_TO_ANALYZE=[configs["model_desc"]]
    
    for model_desc in MODELS_TO_ANALYZE:
        print("Building plot for model: {}".format(model_desc))

        # Summary-metrics
        original_bs=[]
        calibrated_bs_iso=[]

        original_gini=[]
        original_gini_norm=[]
        iso_gini=[]
        iso_gini_norm=[]

        # Error bars
        mean_preds_orig=[]
        mean_preds_iso=[]

        # References axes to interpolate to
        ref_axis_orig=None
        ref_axis_iso=None
        
        for data_split_key,split_key in configs["SPLITS_PER_MODEL"][model_desc]:
            print("Processing split: {}".format(split_key))
            model_col_desc=model_desc
            labels_split=labels_dict[(configs["ml_model_desc"],model_col_desc,data_split_key,split_key)]
            scores_split=scores_dict[(configs["ml_model_desc"],model_col_desc,data_split_key,split_key)]
            labels_cal_split=cal_labels_dict[(configs["ml_model_desc"],model_col_desc,data_split_key,split_key)]
            scores_cal_split=cal_scores_dict[(configs["ml_model_desc"],model_col_desc,data_split_key,split_key)]

            if len(np.unique(labels_split))==1:
                labels_split[0]=1.0
            if len(np.unique(labels_cal_split))==1:
                labels_cal_split[0]=1.0

            cal_model=skiso.IsotonicRegression(out_of_bounds="clip",y_min=0.0,y_max=1.0)
            cal_model.fit(scores_cal_split, labels_cal_split)
            scores_calibrated_iso=cal_model.predict(scores_split)

            frac_pos_orig,mean_pred_orig=rescaled_calibration_curve(labels_split,scores_split,n_bins=20,correct_factor=1.0)

            frac_pos_calibrated_iso,mean_pred_calibrated_iso=rescaled_calibration_curve(labels_split,scores_calibrated_iso,n_bins=20,correct_factor=1.0)

            # Raw scores
            try:
                frac_pos_orig_rs=np.interp(np.arange(0.0,1.01,0.01),mean_pred_orig,frac_pos_orig)
            except:
                continue
                
            ideal_diag=np.arange(0.0,1.01,0.01)
            diff_curve=np.absolute(frac_pos_orig_rs-ideal_diag)
            gini_coeff=skmetrics.auc(np.arange(0.0,1.01,0.01),diff_curve)
            original_gini.append(gini_coeff)

            # Calibrated scores
            try:
                frac_pos_iso_rs=np.interp(np.arange(0.0,1.01,0.01),mean_pred_calibrated_iso,frac_pos_calibrated_iso)
            except:
                continue
                
            ideal_diag=np.arange(0.0,1.01,0.01)
            diff_curve=np.absolute(frac_pos_iso_rs-ideal_diag)
            gini_coeff=skmetrics.auc(np.arange(0.0,1.01,0.01),diff_curve)
            iso_gini.append(gini_coeff)

            if ref_axis_orig is None:
                ref_axis_orig=mean_pred_orig
                ref_axis_iso=mean_pred_calibrated_iso

            mean_preds_orig.append(np.interp(ref_axis_orig,mean_pred_orig,frac_pos_orig))
            mean_preds_iso.append(np.interp(ref_axis_iso,mean_pred_calibrated_iso,frac_pos_calibrated_iso))

            bscore_orig=skmetrics.brier_score_loss(labels_split,scores_split)
            bscore_calibrated_iso=skmetrics.brier_score_loss(labels_split,scores_calibrated_iso)

            original_bs.append(bscore_orig)
            calibrated_bs_iso.append(bscore_calibrated_iso)

        data_out_dict={}

        data_out_dict["perfect_cal_x"]=[0,1]
        data_out_dict["perfect_cal_y"]=[0,1]
    
        data_out_dict["raw_x"]=ref_axis_orig
        data_out_dict["raw_y"]=np.mean(mean_preds_orig,axis=0)
        data_out_dict["raw_fill_min"]=np.maximum(np.mean(mean_preds_orig,axis=0)-np.std(mean_preds_orig,axis=0),0)
        data_out_dict["raw_fill_max"]=np.minimum(np.mean(mean_preds_orig,axis=0)+np.std(mean_preds_orig,axis=0),1)        
        data_out_dict["raw_brier_mean"]=np.mean(original_bs)
        data_out_dict["raw_brier_std"]=np.std(original_bs)
        data_out_dict["raw_gini_mean"]=np.mean(original_gini)
        data_out_dict["raw_gini_std"]=np.std(original_gini)
        data_out_dict["raw_gini_norm_mean"]=np.mean(original_gini_norm)
        data_out_dict["raw_gini_norm_std"]=np.std(original_gini_norm)        

        data_out_dict["iso_x"]=ref_axis_iso
        data_out_dict["iso_y"]=np.mean(mean_preds_iso,axis=0)
        data_out_dict["iso_fill_min"]=np.maximum(np.mean(mean_preds_iso,axis=0)-np.std(mean_preds_iso,axis=0),0)
        data_out_dict["iso_fill_max"]=np.minimum(np.mean(mean_preds_iso,axis=0)+np.std(mean_preds_iso,axis=0),1)        
        data_out_dict["iso_brier_mean"]=np.mean(calibrated_bs_iso)
        data_out_dict["iso_brier_std"]=np.std(calibrated_bs_iso)
        data_out_dict["iso_gini_mean"]=np.mean(iso_gini)
        data_out_dict["iso_gini_std"]=np.std(iso_gini)
        data_out_dict["iso_gini_norm_mean"]=np.mean(iso_gini_norm)
        data_out_dict["iso_gini_norm_std"]=np.std(iso_gini_norm)        
        data_out_dict["num_test_pids"]=len(pred_pids)

        mlhc_io.save_pickle(data_out_dict,os.path.join(configs["data_out_dir"],"{}_{}.pickle".format(task_key, configs["ml_model_desc"])))
    
        
if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--predictions_dir", default="../../data/predictions", help="Which predictions to analyze?")
    
    parser.add_argument("--pid_map_path", default="../../data/exp_design/hirid2_chunking_100.pickle", help="Path of the PID map")
    #parser.add_argument("--pid_map_path", default="../../data/exp_design/umcdb_chunking.pickle", help="Path of the PID map")

    parser.add_argument("--endpoint_path", default="../../data/endpoints/hirid2_endpoints",
                        help="Endpoints to extract the baseline from")

    parser.add_argument("--imputed_path", default="../../data/imputed/impute_hirid2",
                        help="Imputed data to get the S/F ratio from")
    
    parser.add_argument("--temporal_split_path", default="../../data/exp_design/temp_splits_hirid2.pickle", help="Path of temporal split descriptor")
    #parser.add_argument("--temporal_split_path", default="../../data/exp_design/random_splits_umcdb.pickle", help="Path of temporal split descriptor")    
    
    # Output paths
    parser.add_argument("--data_out_dir", default="../../data/evaluation/calibration", help="Path where to store intermediate data")

    # Arguments
    parser.add_argument("--data_mode", default="reduced", help="Should reduced data be used?")
    
    #parser.add_argument("--task_key", default="Label_ExtubationFailureSimple", help="Which label should be evaluated?") 
    parser.add_argument("--task_key", default="Label_WorseStateFromZeroOrOne0To24Hours", help="Which label should be evaluated?")
    #parser.add_argument("--task_key", default="Label_Ventilation0To24Hours", help="Which label should be evaluated?")
    #parser.add_argument("--task_key", default="Label_ReadyExtubate0To24Hours", help="Which label should be evaluated?")           
    
    #parser.add_argument("--model_desc", default="internal_rmsEF", help="Which model type to evaluate?")
    parser.add_argument("--model_desc", default="internal_rmsRF", help="Which model type to evaluate?")
    #parser.add_argument("--model_desc", default="clinical_baseline", help="Which model type to evaluate?")    
    #parser.add_argument("--model_desc", default="internal_rmsVENT", help="Which model type to evaluate?")
    #parser.add_argument("--model_desc", default="internal_rmsREXP", help="Which model type to evaluate?")        
    
    #parser.add_argument("--ml_model_desc", default="lightgbm", help="Which ML model type to evaluate?")
    #parser.add_argument("--ml_model_desc", default="baseline", help="Which ML model type to evaluate?")
    parser.add_argument("--ml_model_desc", default="sf_ratio_baseline", help="Which ML model type to evaluate?")
    #parser.add_argument("--ml_model_desc", default="tree", help="Which ML model type to evaluate?")            
    
    parser.add_argument("--val_type", default="test", help="Which data set to evaluate with?")
    parser.add_argument("--calibration_set", default="val", help="Which data-set should be used for post-hoc calibration of model?")
    
    parser.add_argument("--verbose", default=True, help="Should verbose messages be printed?")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debug mode with fewer patients")
    parser.add_argument("--random_state", type=int, default=2022, help="Random state of RNG")
    parser.add_argument("--small_sample", default=False, action="store_true", help="Small sample?")

    configs=vars(parser.parse_args())

    # configs["ALL_WORK"]=[("lightgbm","internal_rmsEF","temporal_1","temporal_1"),
    #                       ("lightgbm","internal_rmsEF","temporal_2","temporal_2"),
    #                       ("lightgbm","internal_rmsEF","temporal_3","temporal_3"),
    #                       ("lightgbm","internal_rmsEF","temporal_4","temporal_4"),                         
    #                       ("lightgbm","internal_rmsEF","temporal_5","temporal_5")]

    # configs["ALL_WORK"]=[("lightgbm","internal_rmsRF","temporal_1","temporal_1"),
    #                       ("lightgbm","internal_rmsRF","temporal_2","temporal_2"),
    #                       ("lightgbm","internal_rmsRF","temporal_3","temporal_3"),
    #                       ("lightgbm","internal_rmsRF","temporal_4","temporal_4"),                         
    #                       ("lightgbm","internal_rmsRF","temporal_5","temporal_5")]

    # configs["ALL_WORK"]=[("tree","clinical_baseline","temporal_1","temporal_1"),
    #                       ("tree","clinical_baseline","temporal_2","temporal_2"),
    #                       ("tree","clinical_baseline","temporal_3","temporal_3"),
    #                       ("tree","clinical_baseline","temporal_4","temporal_4"),                         
    #                       ("tree","clinical_baseline","temporal_5","temporal_5")]    

    configs["ALL_WORK"]=[("sf_ratio_baseline","internal_rmsRF","temporal_1","temporal_1"),
                         ("sf_ratio_baseline","internal_rmsRF","temporal_2","temporal_2"),
                         ("sf_ratio_baseline","internal_rmsRF","temporal_3","temporal_3"),
                         ("sf_ratio_baseline","internal_rmsRF","temporal_4","temporal_4"),                         
                         ("sf_ratio_baseline","internal_rmsRF","temporal_5","temporal_5")]    

    # configs["ALL_WORK"]=[("lightgbm","internal_rmsVENT","temporal_1","temporal_1"),
    #                       ("lightgbm","internal_rmsVENT","temporal_2","temporal_2"),
    #                       ("lightgbm","internal_rmsVENT","temporal_3","temporal_3"),
    #                       ("lightgbm","internal_rmsVENT","temporal_4","temporal_4"), 
    #                       ("lightgbm","internal_rmsVENT","temporal_5","temporal_5")]

    # configs["ALL_WORK"]=[("lightgbm","internal_rmsREXP","temporal_1","temporal_1"),
    #                       ("lightgbm","internal_rmsREXP","temporal_2","temporal_2"),
    #                       ("lightgbm","internal_rmsREXP","temporal_3","temporal_3"),
    #                       ("lightgbm","internal_rmsREXP","temporal_4","temporal_4"),                         
    #                       ("lightgbm","internal_rmsREXP","temporal_5","temporal_5")]            

    configs["SPLITS_PER_MODEL"]={"internal_rmsRF": [("temporal_1","temporal_1"),
                                                    ("temporal_2","temporal_2"),
                                                    ("temporal_3","temporal_3"),
                                                    ("temporal_4","temporal_4"),                                                      
                                                    ("temporal_5","temporal_5")],
                                 "clinical_baseline": [("temporal_1","temporal_1"),
                                                       ("temporal_2","temporal_2"),
                                                       ("temporal_3","temporal_3"),
                                                       ("temporal_4","temporal_4"),                                                      
                                                       ("temporal_5","temporal_5")],                                 
                                 "internal_rmsVENT": [("temporal_1","temporal_1"),
                                                      ("temporal_2","temporal_2"),
                                                      ("temporal_3","temporal_3"),
                                                      ("temporal_4","temporal_4"),                                                      
                                                      ("temporal_5","temporal_5")],
                                 "internal_rmsREXP": [("temporal_1","temporal_1"),
                                                      ("temporal_2","temporal_2"),
                                                      ("temporal_3","temporal_3"),
                                                      ("temporal_4","temporal_4"),                                                      
                                                      ("temporal_5","temporal_5")],                                                                  
                                 "internal_rmsEF": [("temporal_1","temporal_1"),
                                                    ("temporal_2","temporal_2"),
                                                    ("temporal_3","temporal_3"),
                                                    ("temporal_4","temporal_4"),                                                      
                                                    ("temporal_5","temporal_5")]}
    

    calibration_metrics(configs)
