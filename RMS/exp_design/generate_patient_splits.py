
'''
Generates data splits, guided by the fundamental data split prepared by Alizee already. Thereafter
we just shuffle into train/validation randomly and release these splits to the public.
'''

import random
import csv
import argparse
import ipdb
import os
import pickle
import copy
import sys

import numpy as np
import pandas as pd
import gin

from RMS.utils.io import read_list_from_file, save_pickle
from RMS.utils.admissions import lookup_admission_category, lookup_admission_time 

def execute(configs):
    all_pids=list(map(lambda s: int(float(s)), read_list_from_file(configs["pid_included_list"])))
    print("Included number of PIDs: {}".format(len(all_pids)))
    
    first_date_map=[]

    if configs["endpoint"]=="resp_extval":
        df_patient_full=pd.read_parquet(configs["general_data_table_path"])
        static_pids=list(df_patient_full.admissionid.unique())
    else:
        df_patient_full=pd.read_hdf(configs["general_data_table_path"], mode='r')
        static_pids=list(df_patient_full.PatientID.unique())        

    all_pids=list(set(all_pids).intersection(set(static_pids)))
    print("Number of PIDs after excluding PIDs without static information: {}".format(len(all_pids)))

    # For UMCDB the exact admission times are not available
    if configs["endpoint"]=="resp_extval":
        before_pids=list(filter(lambda elem: lookup_admission_category(elem, df_patient_full)=="2003-2009", all_pids))
        after_pids=list(filter(lambda elem: lookup_admission_category(elem, df_patient_full)=="2010-2016", all_pids))
    else:
        adm_lookup_dict={}
        for pidx,pid in enumerate(all_pids):
            if (pidx+1) % 1000 == 0:
                print("Patient {}/{}".format(pidx+1,len(all_pids)))
            adm_time=lookup_admission_time(pid, df_patient_full)
            first_date_map.append((pid,adm_time))
            adm_lookup_dict[pid]=adm_time

    # Restrict the base cohort to patients between 2010 and 2018
    if configs["restrict_2010_2018"]:
        print("Patients before filtering: {}".format(len(first_date_map)))
        first_date_map=list(filter(lambda item: item[1]>=np.datetime64("2010-01-01T00:00:00.000000000") and item[1]>=np.datetime64("2010-01-01T00:00:00.000000000"), first_date_map))
        print("Patients after filtering: {}".format(len(first_date_map)))

    # Restrict to patients with LOS > 1 day
    elif configs["restrict_los_gt_1_day"]:
        los_pids=list(map(lambda s: int(float(s)), read_list_from_file(configs["los_1_day_list"])))
        print("Number of PIDs before exclusion: {}".format(len(first_date_map)))
        first_date_map=list(filter(lambda item: item[0] in los_pids, first_date_map))
        print("Number of PIDs after exclusion: {}".format(len(first_date_map)))

    # Find matching random sample to LOS 1 > day
    elif configs["match_los_gt_1_day"]:
        los_pids=list(map(lambda s: int(float(s)), read_list_from_file(configs["los_1_day_list"])))
        print("Number of PIDs before exclusion: {}".format(len(first_date_map)))
        first_date_map_tent=list(filter(lambda item: item[0] in los_pids, first_date_map))
        num_pid_exclusions=len(first_date_map_tent)
        first_date_map=random.sample(first_date_map,num_pid_exclusions)
        print("Number of PIDs after exclusion: {}".format(len(first_date_map)))    

    print("Generating temporal/random splits...")

    out_dict={}
    
    if not configs["endpoint"]=="resp_extval":
        all_frame=pd.read_csv(configs["kanonym_pid_list"],sep=',')

        # Get the main split from Alizee's split descriptor
        test_fm=all_frame[all_frame[configs["test_set_col"]]==True]
        train_val_fm=all_frame[all_frame[configs["test_set_col"]]==False]  

        preset_test_pids=list(map(int, test_fm["patientid"].unique()))
        preset_train_val_pids=list(map(int, train_val_fm["patientid"].unique()))
        random_base_pids=preset_test_pids+preset_train_val_pids

    # External validation, fixed random test set
    else:
        random_base_pids=copy.copy(all_pids)
        random.shuffle(random_base_pids)
        preset_test_pids=random_base_pids[:int(0.25*len(random_base_pids))]
        preset_train_val_pids=list(set(all_pids).difference(set(preset_test_pids)))
        
    local_dict={}

    # Produce 5 temporal splits
    if not configs["endpoint"]=="resp_extval":
        for i in range(configs["n_temporal_splits"]):
            local_dict={}
            random.shuffle(preset_train_val_pids)
            local_dict["train"]=preset_train_val_pids[:int(configs["temporal_train_ratio"]*len(preset_train_val_pids))]
            local_dict["val"]=preset_train_val_pids[int(configs["temporal_train_ratio"]*len(preset_train_val_pids)):]
            local_dict["test"]=preset_test_pids
            print("Number of PIDs in temporal split {} train set: {}".format(i,len(local_dict["train"])))             
            print("Number of PIDs in temporal split {} val set: {}".format(i,len(local_dict["val"]))) 
            print("Number of PIDs in temporal split {} test set: {}".format(i,len(local_dict["test"])))
            out_dict["temporal_{}".format(i+1)]=local_dict

    # Produce also 5 random splits as a comparison, for UMCDB these are the only splits produced...
    for i in range(configs["n_random_splits"]):
        local_dict={}
        random.shuffle(preset_train_val_pids)
        local_dict["train"]=preset_train_val_pids[:int(configs["random_train_ratio"]*len(preset_train_val_pids))]
        local_dict["val"]=preset_train_val_pids[int(configs["random_train_ratio"]*len(preset_train_val_pids)):]
        local_dict["test"]=preset_test_pids
        print("Number of PIDs in random split {} train set: {}".format(i,len(local_dict["train"])))
        print("Number of PIDs in random split {} val set: {}".format(i,len(local_dict["val"]))) 
        print("Number of PIDs in random split {} test set: {}".format(i,len(local_dict["test"])))
        out_dict["random_{}".format(i+1)]=local_dict

    if configs["debug_mode"]:
        return

    save_pickle(out_dict, configs["temporal_data_split_binary_path"])

    
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

    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debugging mode, no output to file-system")

    # Choose one for normal splits or external validation splits
    #parser.add_argument("--gin_config", default="./configs/gen_splits.gin", help="Location of GIN config to load, and overwrite the arguments")
    parser.add_argument("--gin_config", default="./configs/gen_splits_extval.gin", help="Location of GIN config to load, and overwrite the arguments")
    
    args=parser.parse_args()
    configs=vars(args)

    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    execute(configs)
