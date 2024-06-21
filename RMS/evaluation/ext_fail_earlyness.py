''' Earlyness analysis for extubation failure'''

import os
import os.path
import argparse
import sys
import glob
import pickle
import ipdb
import random
import copy

import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
import scipy.stats as spstats

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def execute(configs):
    random.seed(2023)
    pred_fs=glob.glob(os.path.join(configs["pred_path"],"reduced",configs["split_key"],
                           configs["task_key"],"batch_*.h5"))
    random.shuffle(pred_fs)

    with open(configs["split_desc"],"rb") as fp:
        test_pids=set(pickle.load(fp)[configs["split_key"]]["test"])

    with open(configs["pid_map"],'rb') as fp:
        pid_map=pickle.load(fp)["pid_to_chunk"]
        test_chunks=set([pid_map[pid] for pid in test_pids])

    true_labels_collect=[]
    scores_collect=[]
    i=0

    n_missed_success=0
    n_caught_success=0
    n_extubations=0
    earlier_times=[]

    patient_early_pids=[]

    early_datetimes=[]
    late_datetimes=[]

    apache_map=pd.read_parquet(configs["apache_map_path"])
    apache_dict=dict(zip(apache_map.meta_group,apache_map.Name))
    df_static=pd.read_hdf(configs["static_path"],mode='r')

    all_test_keys=[]

    event_log_type=[]
    event_log_year=[]
    event_log_month=[]
    event_log_day=[]
    event_log_hour=[]
    event_log_pid=[]

    for fpath in pred_fs:
        chunk_no=int(fpath.split("/")[-1].split("_")[1][:-3])
        if not chunk_no in test_chunks:
            continue

        aux_fpath=os.path.join(configs["pred_path"],"reduced",configs["split_key"],
                               configs["aux_task_key"],"batch_{}.h5".format(chunk_no))

        vent_fpath=os.path.join(configs["pred_path"],"reduced",configs["split_key"],
                                configs["vent_task_key"],"batch_{}.h5".format(chunk_no))        

        i=i+1
        print("{}/{}".format(i,len(test_chunks)))
        hstore=pd.HDFStore(fpath,'r')
        pid_keys=list(hstore.keys())
        pid_keys=set([int(it[2:]) for it in pid_keys])
        test_keys=list(pid_keys.intersection(test_pids))
        if len(test_keys)<=0:
            continue

        all_test_keys.extend(test_keys)
        
        for pid in test_keys:
            df_pid=pd.read_hdf(fpath,'/p{}'.format(pid),mode='r')
            df_pid_aux=pd.read_hdf(aux_fpath,"/p{}".format(pid),mode='r')
            df_pid_vent=pd.read_hdf(vent_fpath,"/p{}".format(pid),mode='r')            
            df_eval=df_pid[df_pid["TrueLabel"].notnull()]

            if df_eval.shape[0]<=0:
                continue

            all_labels=df_pid["TrueLabel"].values
            abs_datetimes=list(df_pid["AbsDatetime"].values)
            all_labels_lst=list(all_labels)
            all_scores=df_pid["PredScore"].values

            # Predicted label is 1 = Extubation success, 0 = Extubation failure
            all_pred_labels=(all_scores<configs["sel_threshold"]).astype(int)
            
            all_rexp=df_pid_aux["TrueLabel"].values
            all_vent=df_pid_vent["TrueLabel"].values

            true_labels_collect.append(df_eval["TrueLabel"].values)
            scores_collect.append(df_eval["PredScore"].values)            

            n_success=np.sum(all_labels==0)
            if n_success<=0:
                continue

            for lidx,label in enumerate(all_labels_lst):

                # Log this hour if the patient is ready to extubate
                if np.isnan(all_rexp[lidx]) and np.isnan(all_vent[lidx]):
                    event_log_type.append("READY_EXTUBATE")
                    event_log_hour.append(pd.Timestamp(abs_datetimes[lidx]).hour)
                    event_log_year.append(pd.Timestamp(abs_datetimes[lidx]).year)
                    event_log_month.append(pd.Timestamp(abs_datetimes[lidx]).month)
                    event_log_day.append(pd.Timestamp(abs_datetimes[lidx]).day)
                    event_log_pid.append(pid)

                if configs["anchor_label"]=="failure":
                    anchor_label_status= label==1 and not all_labels[lidx+1]==1
                elif configs["anchor_label"]=="success":
                    anchor_label_status= label==0 and not all_labels[lidx+1]==0
                    if anchor_label_status:
                        event_log_type.append("SUCCESSFUL_EXTUBATION")
                        event_log_hour.append(pd.Timestamp(abs_datetimes[lidx]).hour)
                        event_log_year.append(pd.Timestamp(abs_datetimes[lidx]).year)
                        event_log_month.append(pd.Timestamp(abs_datetimes[lidx]).month)
                        event_log_day.append(pd.Timestamp(abs_datetimes[lidx]).day)
                        event_log_pid.append(pid)
                
                # Scan to the end of the extubation success/failure label region
                if anchor_label_status:
                        
                    pred_label=all_pred_labels[lidx]
                    n_extubations+=1
                    if pred_label==0:
                        n_missed_success+=1
                    else: # Pred label = 1 (Extubation success)
                        n_caught_success+=1
                        ref_point=lidx
                        bw_point=copy.copy(ref_point)

                        if configs["complex_back_window"]:
                            while bw_point>=5 and np.isnan(all_rexp[(bw_point-5):(bw_point+1)]).all() and \
                                  np.isnan(all_vent[(bw_point-5):(bw_point+1)]).all() and \
                                  np.sum(all_pred_labels[(bw_point-5):(bw_point+1)]==1)>=4:
                                bw_point-=1
                        else:
                            while bw_point>=0 and all_pred_labels[bw_point]==1 and np.isnan(all_rexp[bw_point]) and \
                                  np.isnan(all_vent[bw_point]):

                                # Ignore only in hours where not actually extubated
                                if bw_point<ref_point-12:
                                    event_log_type.append("IGNORED_EXT_SUCCESS_PRED")
                                    event_log_hour.append(pd.Timestamp(abs_datetimes[lidx]).hour)
                                    event_log_year.append(pd.Timestamp(abs_datetimes[lidx]).year)
                                    event_log_month.append(pd.Timestamp(abs_datetimes[lidx]).month)
                                    event_log_day.append(pd.Timestamp(abs_datetimes[lidx]).day)
                                    event_log_pid.append(pid)
                                    
                                bw_point-=1

                        earlier_time=(ref_point-bw_point)*5/60.
                        earlier_times.append(earlier_time)

                        if earlier_time>=configs["early_patient_threshold"]:
                            patient_early_pids.append(pid)
                            early_datetimes.append((pid,abs_datetimes[ref_point],pd.Timestamp(abs_datetimes[ref_point]).hour,earlier_time))
                        else:
                            late_datetimes.append((pd.Timestamp(abs_datetimes[ref_point]).hour,earlier_time))

        if configs["debug_mode"]:
            break

    df_event_log=pd.DataFrame({"pid": event_log_pid, "year": event_log_year, "month": event_log_month,
                              "day": event_log_day, "hour": event_log_hour, "event": event_log_type})
    df_event_log.drop_duplicates(inplace=True)

    print("Number of extubations with status {}: {}".format(configs["anchor_label"],n_extubations))

    if configs["anchor_label"]=="success":
        print("Correctly predicted successes: {}".format(n_caught_success))
        print("Missed successes: {}".format(n_missed_success))
    elif configs["anchor_label"]=="failure":
        print("Missed failures: {}".format(n_caught_success))
        print("Correct failures: {}".format(n_missed_success))        
        
    print("Earlier times [h]: {:.2f} [{:.2f},{:.2f}]".format(np.median(earlier_times),np.percentile(earlier_times,25),
                                                             np.percentile(earlier_times,75)))

    # Save the early date-times to disk
    save_pids=list(map(lambda item: item[0], early_datetimes))
    save_timestamps=list(map(lambda item: item[1], early_datetimes))
    save_early_hours=list(map(lambda item: item[3], early_datetimes))
    df_save=pd.DataFrame({"pid": save_pids, "timestamp": save_timestamps, "hour_offset": save_early_hours})
    df_save.to_parquet(os.path.join(configs["plot_path"],"early_pids_timestamps.parquet"))

    early_dt_hist=dict()
    early_dt_hist_prev=dict()
    late_dt_hist=dict()
    
    for hour in early_datetimes:
        hour_key=hour[2]
        early_hour=hour[3]
        if hour_key not in early_dt_hist:
            early_dt_hist[hour_key]=0
            early_dt_hist_prev[hour_key]=[]
        early_dt_hist[hour_key]+=1
        early_dt_hist_prev[hour_key].append(early_hour)

    for hr_key in early_dt_hist_prev.keys():
        early_dt_hist_prev[hr_key]=(np.percentile(early_dt_hist_prev[hr_key],25),
                                    np.median(early_dt_hist_prev[hr_key]),
                                    np.percentile(early_dt_hist_prev[hr_key],75))
        
    for hour in late_datetimes:
        if hour not in late_dt_hist:
            late_dt_hist[hour]=0
        late_dt_hist[hour]+=1

    print("Number of early events: {}".format(len(patient_early_pids)))
    patient_early_pids=list(set(patient_early_pids))
    print("Unique early pids: {}".format(len(patient_early_pids)))
    apache_group_counts=dict()
    for pid in patient_early_pids:
        df_static_pid=df_static[df_static.PatientID==pid]
        apache_group_entry=df_static_pid.iloc[0]["APACHEPatGroup"]
        if np.isnan(apache_group_entry):
            continue        
        apache_name=apache_dict[int(df_static_pid.iloc[0]["APACHEPatGroup"])]
        if not apache_name in apache_group_counts:
            apache_group_counts[apache_name]=0
        apache_group_counts[apache_name]+=1

    apache_group_counts_control=dict()
    for pid in all_test_keys:
        df_static_pid=df_static[df_static.PatientID==pid]
        apache_group_entry=df_static_pid.iloc[0]["APACHEPatGroup"]
        if np.isnan(apache_group_entry):
            continue
        apache_name=apache_dict[int(df_static_pid.iloc[0]["APACHEPatGroup"])]
        if not apache_name in apache_group_counts_control:
            apache_group_counts_control[apache_name]=0
        apache_group_counts_control[apache_name]+=1        

    print("Early pred APACHE Groups")
    print(apache_group_counts)
    print("All PID APACHE Groups")
    print(apache_group_counts_control)

    plt.hist(earlier_times,bins=50,range=(np.min(earlier_times),np.percentile(earlier_times,99)))
    plt.axvline(np.median(earlier_times),linestyle="-",color="red")
    plt.axvline(np.percentile(earlier_times,25),linestyle="--",color="red")
    plt.axvline(np.percentile(earlier_times,75),linestyle="--",color="red")
    plt.xlabel("Offset first extubation success prediction [h]")
    plt.ylabel("Count")
    plt.savefig(os.path.join(configs["plot_path"],"earliness_ef_tau_{}.pdf".format(configs["sel_threshold"])))
    plt.savefig(os.path.join(configs["plot_path"],"earliness_ef_tau_{}.png".format(configs["sel_threshold"])),dpi=300)
    plt.clf()

    if not configs["threshold_analysis"]:
        sys.exit(0)

    true_labels=np.concatenate(true_labels_collect)
    prevalence=np.sum(true_labels==1)/true_labels.size
    print("Prevalence of EF: {:.2f} %".format(100*prevalence))
    
    scores=np.concatenate(scores_collect)
    score_perc=np.percentile(scores,np.arange(0,100,0.1))

    for score_tau in score_perc:
        pred_labels=(scores>=score_tau).astype(int)
        n_pred_false=np.sum(pred_labels==0)
        n_pred_true=np.sum(pred_labels==1)
        
        n_act_true_pred_false=np.sum((pred_labels==0) & (true_labels==1))
        n_act_false_pred_false=np.sum((pred_labels==0) & (true_labels==0))
        
        n_act_true_pred_true=np.sum((pred_labels==1) & (true_labels==1))
        n_act_true=np.sum(true_labels==1)
        n_act_false=np.sum(true_labels==0)

        prop_detect=n_act_false_pred_false/n_act_false

        recall=n_act_true_pred_true/n_act_true
        prec=n_act_true_pred_true/n_pred_true
        
        for_rate=n_act_true_pred_false/n_pred_false

        #stop_cond=score_tau>=prevalence
        stop_cond=prop_detect>=0.9
        #stop_cond=prop_detect>=0.95
        
        if stop_cond:
            print("Proportion detected Success: {:.2f}".format(prop_detect))
            print("Recall: {:.2f}".format(recall))
            print("Precision: {:.2f}".format(prec))
            print("Threshold: {:.3f}".format(score_tau))
    

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths

    parser.add_argument("--pred_path", default="../../data/predictions",
                        help="Predictions to analyze")

    parser.add_argument("--split_desc", default="../../data/exp_design/temp_splits_hirid2.pickle",
                        help="Split descriptor")

    parser.add_argument("--pid_map", default="../../data/exp_design/hirid2_chunking_100.pickle",
                        help="PID chunking map")

    parser.add_argument("--apache_map_path", default="../../data/misc/apache_metagroup_name.parquet",
                        help="APACHE name map")

    parser.add_argument("--static_path", default="../../data/raw_data/hirid2/static.h5",
                        help="Static patient info path")

    # Output paths
    parser.add_argument("--plot_path", default="../../data/plots/earliness_ef",
                        help="Plot output path")

    # Arguments

    parser.add_argument("--threshold_analysis", default=False, action="store_true", help="Also perform threshold analysis...")

    parser.add_argument("--task_key", default="Label_ExtubationFailureSimple_internal_rmsEF_lightgbm",
                        help="Extubation failure task key")

    parser.add_argument("--aux_task_key", default="Label_ReadyExtubate0To24Hours_internal_rmsREXP_lightgbm",
                        help="Ready to extubate task key")

    parser.add_argument("--vent_task_key", default="Label_Ventilation0To24Hours_internal_rmsVENT_lightgbm",
                        help="Ready to extubate task key")

    parser.add_argument("--complex_back_window", default=False,
                        action="store_true", help="Should a complex back-window calculation be used")

    parser.add_argument("--anchor_label", default="success",
                        help="Anchor event for which to look, can either be failure or success")

    parser.add_argument("--split_key", default="temporal_1", help="Split to analyze in this analysis")

    parser.add_argument("--debug_mode", default=False, action="store_true",
                        help="Debug mode?")

    parser.add_argument("--sel_threshold", default=0.123, help="Selected threshold") # 0.342, 0.269, 0.123

    parser.add_argument("--early_patient_threshold", default=3., help="Long before extubation success threshold")

    configs=vars(parser.parse_args())

    execute(configs)
