
import sys
import os
import os.path
import datetime
import timeit
import random
import gc
import psutil
import csv
import timeit
import time
import argparse
import glob

import ipdb

import pandas as pd
import numpy as np
import scipy as sp

from RMS.utils.io import load_pickle
from RMS.utils.filesystem import delete_if_exist, create_dir_if_not_exist

import label_resp_util as bern_labels


def is_df_sorted(df, colname):
    return (np.array(df[colname].diff().dropna(),dtype=np.float64) >=0).all()


def gen_label(df_pat,df_endpoint,pid=None,configs=None):
    ''' Transform input data-frames to new data-frame with labels'''
    abs_time_col=df_pat[configs["abs_datetime_key"]]
    rel_time_col=df_pat[configs["rel_datetime_key"]]
    patient_col=df_pat[configs["patient_id_key"]]

    if df_pat.shape[0]==0 or df_endpoint.shape[0]==0:
        print("WARNING: Patient {} has no impute data, skipping...".format(pid), flush=True)
        return None

    df_endpoint.set_index(keys="Datetime", inplace=True, verify_integrity=True)
    assert((df_pat.AbsDatetime==df_endpoint.index).all())

    endpoint_status_arr=np.array(df_endpoint.endpoint_status)
    vent_period_arr=np.array(df_endpoint.vent_period)
    ready_ext_arr=np.array(df_endpoint.readiness_ext)
    
    endpoint_status_arr[endpoint_status_arr=="UNKNOWN"]=np.nan
    endpoint_status_arr[endpoint_status_arr=="event_0"]=0
    endpoint_status_arr[endpoint_status_arr=="event_1"]=1
    endpoint_status_arr[endpoint_status_arr=="event_2"]=2
    endpoint_status_arr[endpoint_status_arr=="event_3"]=3
    endpoint_status_arr=endpoint_status_arr.astype(np.float)
    unique_status=np.unique(endpoint_status_arr)

    ext_failure_arr=np.array(df_endpoint["ext_failure"])
    ext_failure_simple_arr=np.array(df_endpoint["ext_failure_simple"])

    for status in unique_status:
        assert(np.isnan(status) or status in [0,1,2,3])

    output_df_dict={}
    output_df_dict[configs["abs_datetime_key"]]=abs_time_col
    output_df_dict[configs["rel_datetime_key"]]=rel_time_col
    output_df_dict[configs["patient_id_key"]]=patient_col

    for (lhours,rhours) in configs["pred_horizons"]:
        from_0_and_1_arr=bern_labels.future_worse_state_from_0_or_1(endpoint_status_arr, lhours, rhours, configs["grid_step_seconds"]) # {0 v 1} -> {>1}
        from_0_arr=bern_labels.future_worse_state_from_0(endpoint_status_arr,lhours,rhours,configs["grid_step_seconds"]) # 0 -> {>0}
        from_1_arr=bern_labels.future_worse_state_from_1(endpoint_status_arr,lhours,rhours,configs["grid_step_seconds"]) # 1 -> {>1}
        from_2_arr=bern_labels.future_worse_state_from_2(endpoint_status_arr, lhours, rhours, configs["grid_step_seconds"]) # 2 -> {3}
        future_vent_arr=bern_labels.future_ventilation(vent_period_arr, lhours, rhours, configs["grid_step_seconds"]) # Ventilation from no ventilation
        future_vent_resource_arr=bern_labels.future_ventilation_resource(vent_period_arr,lhours,rhours,configs["grid_step_seconds"]) # Ventilation from ventilation/no ventilation
        future_ready_ext_arr=bern_labels.future_ready_ext(ready_ext_arr, lhours, rhours, configs["grid_step_seconds"]) # Readiness to extubate from ventilation
        output_df_dict["WorseStateFromZeroOrOne{}To{}Hours".format(lhours, rhours)]=from_0_and_1_arr
        output_df_dict["WorseStateFromZero{}To{}Hours".format(lhours, rhours)]=from_0_arr
        output_df_dict["WorseStateFromOne{}To{}Hours".format(lhours,rhours)]=from_1_arr
        output_df_dict["WorseStateFromTwo{}To{}Hours".format(lhours, rhours)]=from_2_arr
        output_df_dict["Ventilation{}To{}Hours".format(lhours,rhours)]=future_vent_arr
        output_df_dict["VentilationResource{}To{}Hours".format(lhours,rhours)]=future_vent_resource_arr
        output_df_dict["ReadyExtubate{}To{}Hours".format(lhours,rhours)]=future_ready_ext_arr

    future_ready_ext_mc_arr=bern_labels.future_ready_ext_mc(ready_ext_arr, configs["grid_step_seconds"]) # Readiness to extubate with multi-class label
    output_df_dict["ReadyExtubateMulticlass"]=future_ready_ext_mc_arr

    future_vent_mc_arr=bern_labels.future_ventilation_mc(vent_period_arr, configs["grid_step_seconds"])
    output_df_dict["VentilationMulticlass"]=future_vent_mc_arr # Ventilation prediction with a multi-class label
        
    if configs["ext_failure_sample_augment"]:
        for jdx in range(ext_failure_arr.size):
            if np.isfinite(ext_failure_arr[jdx]):
                fail_status=ext_failure_arr[jdx]
                run_idx=jdx
                while run_idx>=0 and jdx-run_idx<int(configs["ext_failure_sample_augment_mins"]/5.) \
                      and vent_period_arr[run_idx]==1:
                    ext_failure_arr[run_idx]=fail_status
                    run_idx-=1
        
    output_df_dict["ExtubationFailure"]=ext_failure_arr

    if configs["ext_failure_sample_augment"]:
        for jdx in range(ext_failure_simple_arr.size):
            if np.isfinite(ext_failure_simple_arr[jdx]):
                fail_status=ext_failure_simple_arr[jdx]
                run_idx=jdx
                while run_idx>=0 and jdx-run_idx<int(configs["ext_failure_sample_augment_mins"]/5.) \
                      and vent_period_arr[run_idx]==1:
                    ext_failure_simple_arr[run_idx]=fail_status
                    run_idx-=1                        

    output_df_dict["ExtubationFailureSimple"]=ext_failure_simple_arr
    
    output_df=pd.DataFrame(output_df_dict)
    return output_df


def label_gen_resp(configs):
    '''Creation of base labels directly defined on the imputed data / endpoints'''
    split_key=configs["split_key"]
    label_base_dir=configs["label_dir"]
    endpoint_base_dir=configs["endpoint_dir"]
    imputed_base_dir=configs["imputed_dir"]
    base_dir=os.path.join(label_base_dir,"reduced",split_key)

    try:
        if not configs["debug_mode"]:
            create_dir_if_not_exist(base_dir,recursive=True)
    except:
        print("WARNING: Race condition when creating directory from different jobs...")

    data_split=load_pickle(configs["temporal_data_split_binary"])[split_key]
    all_pids=data_split["train"]+data_split["val"]+data_split["test"]
    
    if configs["verbose"]:
        print("Number of patient IDs: {}".format(len(all_pids),flush=True))

    batch_map=load_pickle(configs["pid_batch_map_binary"])["chunk_to_pids"]
    batch_idx=configs["batch_idx"]
    
    if not configs["debug_mode"]:
        delete_if_exist(os.path.join(base_dir,"batch_{}.h5".format(batch_idx)))

    pids_batch=batch_map[batch_idx]
    selected_pids=list(set(pids_batch).intersection(all_pids))
    n_skipped_patients=0
    first_write=True
    print("Number of selected PIDs: {}".format(len(selected_pids)),flush=True)

    for pidx,pid in enumerate(selected_pids):
        patient_path=os.path.join(imputed_base_dir, "reduced",split_key,"batch_{}.h5".format(batch_idx))
        cand_files=glob.glob(os.path.join(endpoint_base_dir,split_key,"batch_{}.h5".format(batch_idx)))
        assert(len(cand_files)==1)
        endpoint_path=cand_files[0]
        output_dir=os.path.join(label_base_dir, "reduced", split_key)

        if not os.path.exists(patient_path):
            print("WARNING: Patient {} does not exists, skipping...".format(pid),flush=True)
            n_skipped_patients+=1
            continue

        try:
            df_endpoint=pd.read_hdf(endpoint_path,mode='r', where="PatientID={}".format(pid))
            df_endpoint["Datetime"]=df_endpoint["AbsDatetime"]
        except:
            print("WARNING: Issue while reading endpoints of patient {}".format(pid),flush=True)
            n_skipped_patients+=1
            continue

        df_pat=pd.read_hdf(patient_path,mode='r',where="PatientID={}".format(pid))

        if df_pat.shape[0]==0 or df_endpoint.shape[0]==0:
            print("WARNING: Empty endpoints or empty imputed data in patient {}".format(pid), flush=True)
            n_skipped_patients+=1
            continue

        if not is_df_sorted(df_endpoint, "Datetime"):
            df_endpoint=df_endpoint.sort_values(by="Datetime", kind="mergesort")

        # PRECOND seems fine

        df_label=gen_label(df_pat,df_endpoint,pid=pid,configs=configs)

        if df_label is None:
            print("WARNING: Label could not be created for PID: {}".format(pid),flush=True)
            n_skipped_patients+=1
            continue

        assert(df_label.shape[0]==df_pat.shape[0])
        output_path=os.path.join(output_dir,"batch_{}.h5".format(batch_idx))

        if first_write:
            append_mode=False
            open_mode='w'
        else:
            append_mode=True
            open_mode='a'

        if not configs["debug_mode"]:
            df_label.to_hdf(output_path,configs["label_dset_id"],complevel=configs["hdf_comp_level"],complib=configs["hdf_comp_alg"], format="table",
                            append=append_mode, mode=open_mode, data_columns=["PatientID"])

        gc.collect()
        first_write=False

        if (pidx+1)%100==0 and configs["verbose"]:
            print("Progress for batch {}: {:.2f} %".format(batch_idx, (pidx+1)/len(selected_pids)*100),flush=True)
            print("Number of skipped patients: {}".format(n_skipped_patients))
