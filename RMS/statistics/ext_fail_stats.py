''' Statistics about extubation failure'''

import os
import os.path
import argparse
import ipdb
import glob
import random

import numpy as np
import pandas as pd

def analyze_ext_fail(batch_fs, dset_name):
    cnt_with_ef=0
    cnt_no_ef=0
    cnt_no_ext=0
    cnt_with_ext=0

    cnt_ext_success=0
    cnt_ext_failure=0
    
    for f_idx,f_batch in enumerate(batch_fs):
        print("Batch {}/{}".format(f_idx+1,len(batch_fs)))
        df_labels=pd.read_hdf(f_batch,mode='r')
        df_labels["PatientID"]=df_labels["PatientID"].astype('category')
        all_pids=list(df_labels["PatientID"].unique())

        for pid in all_pids:
            df_pid=df_labels[df_labels.PatientID==pid]

            failure_vect=df_pid["ExtubationFailureSimple"].values

            if np.sum(failure_vect==1)>0:
                cnt_with_ef+=1
            else:
                cnt_no_ef+=1

            if np.sum(np.isfinite(failure_vect))>0:
                cnt_with_ext+=1
            else:
                cnt_no_ext+=1

            ext_periods=df_pid[df_pid.ExtubationFailureSimple.notnull()]

            if ext_periods.shape[0]>0:
                ext_periods_begin=ext_periods[~(ext_periods.RelDatetime.diff()==300)]
                ext_periods_vals=ext_periods_begin["ExtubationFailureSimple"].values
                cnt_ext_success+=np.sum(ext_periods_vals==0)
                cnt_ext_failure+=np.sum(ext_periods_vals==1)
            
    print("{} - Patients with EF: {}, no EF: {}".format(dset_name,cnt_with_ef,cnt_no_ef))
    print("{} - Patients with Ext: {}, no Ext: {}".format(dset_name,cnt_with_ext,cnt_no_ext))
    print("{} - Ext. success count: {}, Ext fail count: {}".format(dset_name,cnt_ext_success,cnt_ext_failure))    
    

def execute(configs):

    label_f_hirid=glob.glob(os.path.join(configs["hirid_label_dir"],"reduced",
                                         "temporal_1","batch*.h5"))

    label_f_umcdb=glob.glob(os.path.join(configs["umc_label_dir"],"reduced",
                                         "random_1","batch*.h5"))

    if configs["small_sample"]:
        random.shuffle(label_f_hirid)
        random.shuffle(label_f_umcdb)
        label_f_hirid=label_f_hirid[:5]
        label_f_umcdb=label_f_umcdb[:5]

    analyze_ext_fail(label_f_hirid,"HiRID")
    analyze_ext_fail(label_f_umcdb,"UMCDB")

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths

    parser.add_argument("--hirid_label_dir", default="../../data/labels/hirid2_labels",
                        help="UMC labels")    
    
    parser.add_argument("--umc_label_dir", default="../../data/labels/umcdb_labels_OLD",
                        help="UMC labels")

    # Output paths

    # Arguments
    parser.add_argument("--small_sample", default=False, action="store_true",
                        help="Small sample")

    configs=vars(parser.parse_args())

    execute(configs)
