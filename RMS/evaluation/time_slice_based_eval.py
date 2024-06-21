'''
Generate tables of time-slice based results for a set of prediction results
'''
import argparse
import os
import os.path
import math
import itertools
import ipdb
import pickle
import warnings

import pandas as pd
import numpy as np
import scipy
import gin

import sklearn.metrics as skmetrics

from RMS.utils.io import load_pickle


def subsample_cohort(base_pids, df_static, apache_map, subsample_desc):
    ''' Subsamples prediction PIDs'''
    apache_dict=dict(zip(apache_map.Name,apache_map.meta_group))

    # Select the correct gender group
    if "FEMALE" in subsample_desc:
        df_selected=df_static[df_static.Sex=="F"]    
    elif "MALE" in subsample_desc:
        df_selected=df_static[df_static.Sex=="M"]
    else:
        df_selected=df_static

    # Select the correct age group
    if "AGE" in subsample_desc:
        age_str=subsample_desc.split("_")[1:]
        if age_str[0]=="16" and age_str[1]=="30":
            df_selected=df_selected[(df_selected.Age>=16) & (df_selected.Age<=30)]
        elif age_str[0]=="31" and age_str[1]=="45":
            df_selected=df_selected[(df_selected.Age>=31) & (df_selected.Age<=45)]
        elif age_str[0]=="46" and age_str[1]=="65":
            df_selected=df_selected[(df_selected.Age>=46) & (df_selected.Age<=65)]
        elif age_str[0]=="66" and age_str[1]=="80":
            df_selected=df_selected[(df_selected.Age>=66) & (df_selected.Age<=80)]
        elif age_str[0]=="81" and age_str[1]=="100":
            df_selected=df_selected[(df_selected.Age>=81) & (df_selected.Age<=100)]
        else:
            ipdb.set_trace()

    elif "APACHE" in subsample_desc:
        assert "APACHE" in subsample_desc

        group_desc=subsample_desc[7:]
        if group_desc=="cardio_surgical":
            df_selected=df_selected[df_selected.APACHEPatGroup==apache_dict["Cardiovascular/vascular surgical"]]
        elif group_desc=="cardio_medical":
            df_selected=df_selected[df_selected.APACHEPatGroup==apache_dict["Cardiovascular"]]
        elif group_desc=="resp_surgical":
            df_selected=df_selected[df_selected.APACHEPatGroup==apache_dict["Respiratory surgical"]]
        elif group_desc=="resp_medical":
            df_selected=df_selected[df_selected.APACHEPatGroup==apache_dict["Respiratory"]]
        elif group_desc=="gastro_surgical":
            df_selected=df_selected[df_selected.APACHEPatGroup==apache_dict["Gastrointestinal surgical"]]
        elif group_desc=="gastro_medical":
            df_selected=df_selected[df_selected.APACHEPatGroup==apache_dict["Gastrointestinal"]]
        elif group_desc=="neuro_surgical":
            df_selected=df_selected[df_selected.APACHEPatGroup==apache_dict["Neurologic surgical"]]
        elif group_desc=="neuro_medical":
            df_selected=df_selected[df_selected.APACHEPatGroup==apache_dict["Neurologic"]]
        elif group_desc=="trauma_surgical":
            df_selected=df_selected[df_selected.APACHEPatGroup==apache_dict["Trauma surgical"]]
        elif group_desc=="trauma_medical":
            df_selected=df_selected[df_selected.APACHEPatGroup==apache_dict["Trauma"]]
        elif group_desc=="metabolic_medical":
            df_selected=df_selected[df_selected.APACHEPatGroup==apache_dict["Metabolic/Endocrinology"]]
        else:
            ipdb.set_trace()

    filter_pids=list(df_selected["PatientID"].unique())
    filtered_pids=list(filter(lambda pid: pid in filter_pids, base_pids))
    return filtered_pids


def prec_at_recall_fixed(recalls, precs, level=None):
    recalls=list(recalls)
    precs=list(precs)
    assert len(recalls)==len(precs)
    for idx,recall in enumerate(recalls):
        if recall>=level/100:
            return 100*precs[idx]
    return None

def custom_roc_curve(labels, scores):
    ''' A custom ROC curve with a large number of thresholds'''
    n_thresholds=10000
    perc_range=np.flip(np.linspace(0,100,n_thresholds))
    fpr_out=[]
    tpr_out=[]
    taus=[]
    for ts in scores:
        #ts=np.percentile(scores,perc_ts)
        taus.append(ts)
        pred_labels=(scores>=ts).astype(int)
        fpr=1-np.sum((pred_labels==0)&(labels==0))/np.sum(labels==0)
        tpr=np.sum((pred_labels==1)&(labels==1))/np.sum(labels==1)
        fpr_out.append(fpr)
        tpr_out.append(tpr)
    
    return (np.array(fpr_out),np.array(tpr_out),np.array(taus))
    


def custom_pr_curve(labels, scores):
    ''' A custom PR curve with a large number of thresholds'''
    n_thresholds=10000
    perc_range=np.linspace(0,100,n_thresholds)
    precs_out=[]
    recs_out=[]
    taus=[]
    for ts in scores:
        #ts=np.percentile(scores,perc_ts)
        taus.append(ts)
        pred_labels=(scores>=ts).astype(int)
        rec=np.sum((pred_labels==1)&(labels==1))/np.sum(labels==1)
        prec=np.sum((pred_labels==1)&(labels==1))/np.sum(pred_labels==1)
        precs_out.append(prec)
        recs_out.append(rec)
    return (np.array(precs_out),np.array(recs_out),np.array(taus))
    


def corrected_pr_curve(labels, scores, correct_factor=None, custom_curve=False):
    ''' Returns a collection of metrics'''
    taus=[]
    tps=[]
    fps=[]
    npos=np.sum(labels==1.0)    

    if custom_curve:
        threshold_set=np.copy(scores)
    else:
        threshold_set=np.arange(0.0,1.001,0.001)
    
    for tau in threshold_set:
        #tau=np.percentile(scores,perc_ts)
        der_labels=(scores>=tau).astype(int)
        taus.append(tau)
        tp=np.sum((labels==1.0) & (der_labels==1.0))
        fp=np.sum((labels==0.0) & (der_labels==1.0))
        tps.append(tp)
        fps.append(fp)

    tps=np.array(tps)
    fps=np.array(fps)
    taus=np.array(taus)

    recalls=tps/npos
    precisions=tps/(tps+correct_factor*fps)
    precisions[np.isnan(precisions)]=1.0
        
    return (precisions, recalls, taus)


def execute(configs):

    n_skipped_patients=0
    scores_dict={}
    labels_dict={}

    df_out_dict= { 
        "test_set_pids": [],
        "test_set_samples": [],
        "auroc": [],
        "auprc": [],
        "prec_at_20": [],
        "prec_at_80": [],
        "precision": [],
        "recall": [],
        "fpr": [],
        "label_prevalence": [],
        "task": [],
        "database": [],
        "split": []
    }

    curve_dict= {
        "roc_curves": {},
        "pr_curves": {}
    }

    work_idx=1
    all_work=configs["task_keys"]

    if configs["restrict_tis_hours"] is not None:
        print("Restricting evaluation to {} [h] into stay".format(configs["restrict_tis_hours"]))

    if configs["autosense_eval_hour"]:
        print("Auto-restricting evaluation hour")

    track_largest_score_ef=-np.inf
    track_smallest_score_sf_ratio=np.inf
    track_largest_score_sf_ratio=-np.inf
        
    for split_key in configs["eval_splits"]:
        bern_split_key=split_key[0]
        umc_split_key=split_key[1]
        
        print("Processing split: {}".format(split_key))
        
        for label_key,custom_str,database,correct_curve,subsample_desc in all_work:
            print("Processing model {}/{}".format(work_idx,len(all_work)))

            if configs["autosense_eval_hour"]:
                eval_point=int(label_key.split("_")[2][2:-5])

            work_idx+=1

            if custom_str is not None:
                df_out_dict["task"].append("{}_{}".format(label_key,custom_str))
            elif subsample_desc is not None:
                df_out_dict["task"].append("{}_{}".format(label_key,subsample_desc))
            else:
                df_out_dict["task"].append(label_key)

            df_out_dict["database"].append(database)

            df_out_dict["split"].append(split_key)
            cum_pred_scores=[]
            cum_labels=[]

            if database=="hirid":
                data_split=load_pickle(configs["bern_temporal_split_path"])[bern_split_key]
                batch_map=load_pickle(configs["hirid_pid_map_path"])["pid_to_chunk"]            
            elif database=="umcdb":
                data_split=load_pickle(configs["umc_temporal_split_path"])[umc_split_key]
                batch_map=load_pickle(configs["umc_pid_map_path"])["pid_to_chunk"]

            pred_pids=data_split["test"]

            if subsample_desc is not None:
                df_static=pd.read_hdf(configs["static_path"],mode='r')
                apache_map=pd.read_parquet(configs["apache_map_path"])
                pred_pids=subsample_cohort(pred_pids,df_static,apache_map,subsample_desc)
            
            n_test_pids=0
            nan_pred_cnt=0
            coh_cnt=0
            incoh_cnt=0

            if configs["internal_mode"] or "_internal_" in label_key or "_val_" in label_key:
                output_dir=os.path.join(configs["bern_predictions_dir"],"reduced",bern_split_key,label_key)
            elif "_retrain_" in label_key:
                output_dir=os.path.join(configs["bern_predictions_dir"],"reduced",umc_split_key,label_key) 

            ep_dir=os.path.join(configs["endpoint_path"],bern_split_key)
            imputed_dir=os.path.join(configs["imputed_path"],"reduced",bern_split_key)

            print("Loading patients...")

            pred_frame_buffer=dict()
            
            for pidx,pid in enumerate(pred_pids):
                if (pidx+1)%100==0 and configs["verbose"]:
                    print("{}/{}".format(pidx+1,len(pred_pids)))
                if pidx>=500 and configs["debug_mode"]:
                    break
                batch_pat=batch_map[pid]
                try:
                    search_f=os.path.join(output_dir,"batch_{}.h5".format(batch_pat))

                    if "one_minus" in label_key:
                        if batch_pat in pred_frame_buffer:
                            df_pred=pred_frame_buffer[batch_pat]
                        else:
                            df_pred=pd.read_hdf(search_f,"/data", mode='r')
                            pred_frame_buffer[batch_pat]=df_pred
                            
                        df_pred=df_pred[df_pred.PatientID==pid]
                        ref_output_dir=os.path.join(configs["bern_predictions_dir"],"reduced",bern_split_key,"Label_WorseStateFromZeroOrOne0To24Hours_internal_rmsRF_lightgbm")
                        ref_search_f=os.path.join(ref_output_dir,"batch_{}.h5".format(batch_pat))
                        df_ref_pred=pd.read_hdf(ref_search_f,"/p{}".format(pid), mode='r')
                        assert df_ref_pred.shape[0]==df_pred.shape[0]

                        with warnings.catch_warnings():
                            warnings.simplefilter(action='ignore')
                            df_pred["TrueLabel"]=df_ref_pred["TrueLabel"].values
                            df_pred["RelDatetime"]=df_ref_pred["RelDatetime"].values
                            df_pred=df_pred.rename(columns={"Prediction": "PredScore"})
                    else:
                        df_pred=pd.read_hdf(search_f,"/p{}".format(pid), mode='r')

                    if custom_str=="custom_threshold":
                        df_ep=pd.read_hdf(os.path.join(ep_dir,"batch_{}.h5".format(batch_pat)),
                                          mode='r',where="PatientID={}".format(pid))
                        df_ep=df_ep[pd.notna(df_pred["TrueLabel"]) & pd.notna(df_pred["PredScore"])]
                    elif custom_str=="custom_sf_ratio":
                        df_imputed=pd.read_hdf(os.path.join(imputed_dir,"batch_{}.h5".format(batch_pat)),
                                               mode='r',where="PatientID={}".format(pid))
                        df_imputed=df_imputed[pd.notna(df_pred["TrueLabel"]) & pd.notna(df_pred["PredScore"])][["AbsDatetime","RelDatetime","vm20","vm58"]]
                        
                    if "Multiclass" in label_key:
                        df_pred=df_pred[pd.notna(df_pred["TrueLabel"]) & pd.notna(df_pred["PredScore_0"])]
                    else:
                        df_pred=df_pred[pd.notna(df_pred["TrueLabel"]) & pd.notna(df_pred["PredScore"])]


                except (KeyError, FileNotFoundError) as exc:
                    n_skipped_patients+=1
                    continue

                n_test_pids+=1

                if configs["restrict_tis_hours"] is not None:
                    df_pred=df_pred[df_pred["RelDatetime"]==3600*configs["restrict_tis_hours"]]

                if configs["autosense_eval_hour"]:
                    df_pred=df_pred[df_pred["RelDatetime"]==3600*eval_point]

                if df_pred.shape[0]==0:
                    continue

                if custom_str is None:
                    if "Multiclass" in label_key:
                        pred_scores=np.array(df_pred["PredScore_1"])+np.array(df_pred["PredScore_2"])
                    else:
                        pred_scores=np.array(df_pred["PredScore"])

                elif custom_str=="custom_threshold":
                    raw_scores=df_ep["ext_ready_violation_score"].values
                    pred_scores=raw_scores/35.  # We use that 35 is an upper bound to the largest prediction score
                    assert np.sum((pred_scores<0) | (pred_scores>1))==0

                elif custom_str=="custom_sf_ratio":
                    spo2_vals=df_imputed["vm20"].values
                    fio2_vals=df_imputed["vm58"].values
                    sf_ratios=spo2_vals/fio2_vals

                    # Normalize to unit interval and invert the scores
                    pred_scores=(sf_ratios-0.13)/(4.77-0.13)
                    pred_scores=1-pred_scores
                    assert np.sum((pred_scores<0) | (pred_scores>1))==0

                true_labels=np.array(df_pred["TrueLabel"])

                if np.sum(np.isnan(pred_scores))==0 and np.sum(np.isnan(true_labels))==0:
                    cum_pred_scores.append(pred_scores)
                    cum_labels.append(true_labels)
                else:
                    nan_pred_cnt+=1

            df_out_dict["test_set_pids"].append(n_test_pids)
            n_test_samples=np.concatenate(cum_pred_scores).size
            df_out_dict["test_set_samples"].append(n_test_samples)
            raw_scores=np.concatenate(cum_pred_scores)
            raw_labels=np.concatenate(cum_labels)

            print("Number of scores: {}".format(len(raw_scores)))

            assert(np.sum(np.isnan(raw_labels))==0)
            prevalence=np.sum(raw_labels==1.0)/raw_labels.size
            df_out_dict["label_prevalence"].append(prevalence)

            if not correct_curve:
                reference_prevalence=prevalence
                print("Reference prevalence: {:.3f}".format(reference_prevalence))

            if configs["invert_scores"]:
                raw_labels=1.0-raw_labels
                raw_scores=1.0-raw_scores

            if "Multiclass" in label_key and configs["conf_matrix_eval"]:
                conf_matrix=skmetrics.confusion_matrix(raw_labels,raw_scores)
            else:

                if configs["custom_roc_pr_curves"]:
                    fpr_model,tpr_model,_=custom_roc_curve(raw_labels, raw_scores)
                else:
                    fpr_model,tpr_model,_=skmetrics.roc_curve(raw_labels,raw_scores,pos_label=1)

                tpr_model=tpr_model[np.argsort(fpr_model)]
                fpr_model=fpr_model[np.argsort(fpr_model)]                

                if custom_str is not None:
                    rev_label_key="{}_{}".format(label_key,custom_str)
                else:
                    rev_label_key=label_key
                    
                curve_dict["roc_curves"][(database,rev_label_key,split_key)]={"fpr": fpr_model, "tpr": tpr_model}
                print("Scores Mean: {:.3f}, {:.3f}".format(np.mean(raw_scores), np.std(raw_scores)))
                auroc=skmetrics.auc(fpr_model,tpr_model)
                df_out_dict["auroc"].append(auroc)

            df_out_dict["fpr"].append(np.nan)
            df_out_dict["precision"].append(np.nan)
            df_out_dict["recall"].append(np.nan)

            if correct_curve:
                correction_factor=(1/reference_prevalence-1)/(1/prevalence-1)
                print("Target prevalence: {:.3f}".format(prevalence))
                print("Correction factor: {:.3f}".format(correction_factor))
                precs_model,recalls_model,_=corrected_pr_curve(raw_labels,raw_scores,correct_factor=correction_factor,
                                                               custom_curve=configs["custom_roc_pr_curves"])
            else:

                if configs["custom_roc_pr_curves"]:
                    precs_model,recalls_model,_=custom_pr_curve(raw_labels, raw_scores)
                else:
                    precs_model,recalls_model,_=skmetrics.precision_recall_curve(raw_labels,raw_scores,pos_label=1)

            precs_model=precs_model[np.argsort(recalls_model)]
            recalls_model=recalls_model[np.argsort(recalls_model)]

            prec_at_80=prec_at_recall_fixed(recalls_model, precs_model, level=80)
            prec_at_20=prec_at_recall_fixed(recalls_model, precs_model, level=20)
            df_out_dict["prec_at_80"].append(prec_at_80)
            df_out_dict["prec_at_20"].append(prec_at_20)

            auprc=skmetrics.auc(recalls_model,precs_model)

            df_out_dict["auprc"].append(auprc)
            curve_dict["pr_curves"][(database,rev_label_key,split_key)]={"recalls": recalls_model, "precs": precs_model}

    result_frame=pd.DataFrame(df_out_dict)

    if configs["restrict_tis_hours"] is not None:
        result_frame.to_csv(os.path.join(configs["eval_table_dir"], "task_results_TIS{}.tsv".format(configs["restrict_tis_hours"])),sep='\t',index=False)
        with open(os.path.join(configs["eval_table_dir"], "raw_results_TIS{}.pickle".format(configs["restrict_tis_hours"])),'wb') as fp:
            pickle.dump(curve_dict,fp)
    else:
        result_frame.to_csv(os.path.join(configs["eval_table_dir"], "task_results.tsv"),sep='\t',index=False)
        with open(os.path.join(configs["eval_table_dir"], "raw_results.pickle"),'wb') as fp:
            pickle.dump(curve_dict,fp)

@gin.configurable
def parse_gin_args(old_configs,gin_configs=None):
    gin_configs=gin.query_parameter("parse_gin_args.gin_configs")
    for k in old_configs.keys():
        if old_configs[k] is not None:
            gin_configs[k]=old_configs[k]
    gin.bind_parameter("parse_gin_args.gin_configs",gin_configs)
    return gin_configs
 
if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--verbose", default=None, action="store_true", help="Should verbose messages be printed?")
    parser.add_argument("--debug_mode", default=None, action="store_true", help="Debug mode with fewer patients")

    parser.add_argument("--gin_config", default="./configs/eval_rf.gin", help="GIN config to use")    
    #parser.add_argument("--gin_config", default="./configs/eval_rf_extval.gin", help="GIN config to use")
    
    #parser.add_argument("--gin_config", default="./configs/eval_ef_extval.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/eval_eflite_extval.gin", help="GIN config to use") 
    #parser.add_argument("--gin_config", default="./configs/eval_ef.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/eval_ef_cohorts.gin", help="GIN config to use")    

    #parser.add_argument("--gin_config", default="./configs/eval_vent.gin", help="GIN config to use")

    #parser.add_argument("--gin_config", default="./configs/eval_rexp.gin", help="GIN config to use")         
    
    #parser.add_argument("--gin_config", default="./configs/eval_ext_fail_prefix.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/eval_ext_fail_prefix_no_pharma.gin", help="GIN config to use")    
    #parser.add_argument("--gin_config", default="./configs/eval_ext_fail_prefix_val.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/eval_ext_fail_prefix_val_no_pharma.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/eval_ef.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/eval_ef_training_size.gin", help="GIN config to use")    
    #parser.add_argument("--gin_config", default="./configs/eval_resp_fail_single_model.gin", help="GIN config to use") 
    #parser.add_argument("--gin_config", default="./configs/eval.gin", help="GIN config to use")       
    #parser.add_argument("--gin_config", default="./configs/eval_umcdb.gin", help="GIN config to use")
    
    parser.add_argument("--restrict_tis_hours", default=None, type=int, help="Restrict evaluation to a certain hour")

    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    execute(configs)
