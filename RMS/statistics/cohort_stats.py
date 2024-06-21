''' Cohort statistics to fill table 1 of the paper'''

import os
import os.path
import pickle
import ipdb
import argparse
import glob
import random

import pandas as pd
import numpy as np

def iqr(pd_col):
    quartiles = pd_col.quantile([0.25, 0.75])
    iqr = quartiles[0.75] - quartiles[0.25]
    return iqr,quartiles[0.25],quartiles[0.75]

def execute(configs):

    with open(configs["hirid_split_desc"],'rb') as fp:
        hirid_desc=pickle.load(fp)

    with open(configs["umc_split_desc"],'rb') as fp:
        umc_desc=pickle.load(fp)

    hirid_n_dev=len(hirid_desc["temporal_1"]["train"])+len(hirid_desc["temporal_1"]["val"])

    hirid_n_train=len(hirid_desc["temporal_1"]["train"])
    hirid_n_val=len(hirid_desc["temporal_1"]["val"])
    print("Number of patients in HIRID train/val sets: {}/{}".format(hirid_n_train,hirid_n_val))
    
    hirid_n_test=len(hirid_desc["temporal_1"]["test"])

    umc_n_dev=len(umc_desc["random_1"]["train"])+len(umc_desc["random_1"]["val"])
    umc_n_test=len(umc_desc["random_1"]["test"])

    all_hirid_patients=hirid_desc["temporal_1"]["train"]+hirid_desc["temporal_1"]["val"]+hirid_desc["temporal_1"]["test"]
    all_umc_patients=umc_desc["random_1"]["train"]+umc_desc["random_1"]["val"]+umc_desc["random_1"]["test"]

    print("Number of patients in HIRID dev/test sets: {}/{}".format(hirid_n_dev,hirid_n_test))
    print("Number of patients in UMCDB dev/test sets: {}/{}".format(umc_n_dev,umc_n_test))
    
    hirid_static=pd.read_hdf(configs["hirid_static_data"])
    umc_static=pd.read_hdf(configs["umc_static_data"])
    umc_static_misc=pd.read_parquet(configs["umc_static_misc_data"])

    hirid_static=hirid_static[hirid_static.PatientID.isin(all_hirid_patients)]
    umc_static=umc_static[umc_static.PatientID.isin(all_umc_patients)]
    umc_static.drop_duplicates(subset=["PatientID"],inplace=True)

    umc_static_misc=umc_static_misc[umc_static_misc.patientid.isin(all_umc_patients)]
    umc_static_misc.drop_duplicates(subset=["patientid"],inplace=True)

    hirid_med_age=hirid_static["Age"].median()
    hirid_iqr_age=iqr(hirid_static["Age"])

    umc_med_age=umc_static["Age"].median()
    umc_iqr_age=iqr(umc_static["Age"])

    print("HiRID-II age: {:.2f} [{:.2f},{:.2f}]".format(hirid_med_age,hirid_iqr_age[1],hirid_iqr_age[2]))
    print("UMCDB age: {:.2f} [{:.2f},{:.2f}]".format(umc_med_age,umc_iqr_age[1],umc_iqr_age[2]))
    
    print("HIRID-II sex distribution: {}".format(hirid_static["Sex"].value_counts()))
    print("UMCDB sex distribution: {}".format(umc_static["Sex"].value_counts()))  

    print("HIRID-II emergency distribution: {}".format(hirid_static["Emergency"].value_counts()))
    print("UMCDB emergency distribution: {}".format(umc_static["Emergency"].value_counts()))    

    # Length of stay [days]
    hirid_imputed_batches=glob.glob(os.path.join(configs["hirid_imputed_data"],"reduced","temporal_1","batch_*.h5"))
    umc_imputed_batches=glob.glob(os.path.join(configs["umc_imputed_data"],"reduced","random_1","batch_*.h5"))

    if configs["small_sample"]:
        random.shuffle(hirid_imputed_batches)
        random.shuffle(umc_imputed_batches)
        hirid_imputed_batches=hirid_imputed_batches[:5]
        umc_imputed_batches=umc_imputed_batches[:5]

    # UMCDB mortality analysis
    all_umcdb_pids=list(umc_static_misc["patientid"].unique())
    cnt_umcdb_discharge_dead=0
    for pid in all_umcdb_pids:
        df_pid=umc_static_misc[umc_static_misc["patientid"]==pid]
        df_dead=df_pid[df_pid.dateofdeath.notnull()]
        if df_dead.shape[0]>0:
            icu_discharge=df_pid["dischargedat"].max()
            death_stamp=df_dead["dateofdeath"].min()
            if death_stamp<=icu_discharge:
                cnt_umcdb_discharge_dead+=1
    
    print("HIRID-II mortality distribution: {}".format(hirid_static["Discharge"].value_counts()))
    print("UMCDB discharge mortality: {:.2f} %".format(100*cnt_umcdb_discharge_dead/len(all_umcdb_pids)))
        
    los_hirid=[]
    
    for bix,batch_f in enumerate(hirid_imputed_batches):
        print("Processing batch: {}/{}".format(bix+1,len(hirid_imputed_batches)))
        df_batch=pd.read_hdf(batch_f)
        pids_batch=df_batch["PatientID"].unique()
        for pid in pids_batch:
            df_pid=df_batch[df_batch["PatientID"]==pid]
            los=df_pid["RelDatetime"].max()/3600./24.
            los_hirid.append(los)

    print("HiRID-II, LOS median: {:.2f}, Q1: {:.2f}, Q3: {:.2f}".format(np.median(los_hirid), np.percentile(los_hirid,25),
                                                                        np.percentile(los_hirid,75)))

    los_umc=[]
    
    for bix,batch_f in enumerate(umc_imputed_batches):
        print("Processing batch: {}/{}".format(bix+1,len(umc_imputed_batches)))
        df_batch=pd.read_hdf(batch_f)
        pids_batch=df_batch["PatientID"].unique()
        for pid in pids_batch:
            df_pid=df_batch[df_batch["PatientID"]==pid]
            los=df_pid["RelDatetime"].max()/3600./24.
            los_umc.append(los)

    print("UMCDB, LOS median: {:.2f}, Q1: {:.2f}, Q3: {:.2f}".format(np.median(los_umc), np.percentile(los_umc,25),
                                                                     np.percentile(los_umc,75)))





if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--hirid_split_desc", default="../../data/exp_design/temp_splits_hirid2.pickle",
                        help="HiRID split descriptor")

    parser.add_argument("--umc_split_desc", default="../../data/exp_design/random_splits_umcdb.pickle",
                        help="UMCDB split descriptor")

    parser.add_argument("--hirid_static_data", default="../../data/raw_data/hirid2/static.h5",
                        help="Path to HiRID static data")

    parser.add_argument("--umc_static_data", default="../../data/imputed/noimpute_umcdb_OLD/reduced/random_1/static.h5",
                        help="Path to UMCDB static data")
    parser.add_argument("--umc_static_misc_data", default="../../data/raw_data/umcdb/admissions.parquet",
                        help="Path to UMCDB static data")    

    parser.add_argument("--hirid_imputed_data", default="../../data/imputed/noimpute_hirid2",
                        help="Path to HiRID imputed data")

    parser.add_argument("--umc_imputed_data", default="../../data/imputed/noimpute_umcdb_OLD",
                        help="Path to UMCDB imputed data")

    # Output paths

    # Arguments
    parser.add_argument("--small_sample", default=False, action="store_true",
                        help="Use a small sample to estimate LOS")

    configs=vars(parser.parse_args())

    execute(configs)
