''' Saves a list of PIDs to be included as a basis for split generation'''

import argparse
import glob
import os
import os.path
import gin
import pickle
import ipdb
import tqdm

import pandas as pd

from RMS.utils.io import write_list_to_file

def execute(configs):

    # Respiratory failure, can compute endpoints on all IDs in merged
    if configs["endpoint"]=="resp":
        uniq_pids=[]
        for idx,fpath in enumerate(glob.glob(os.path.join(configs["merged_path"], "reduced_fmat_*.h5"))):
            df=pd.read_hdf(fpath,mode='r')
            print("Index: {}".format(idx))
            uniq_pids.extend(list(df.PatientID.unique()))
            if configs["debug_mode"]:
                break
        print("PIDs in merged: {}".format(len(set(uniq_pids))))

        if configs["restrict_k_anonym"]:
            anon_desc=pd.read_csv(configs["kanonym_pid_list"],sep=',')
            anon_desc=anon_desc[anon_desc[configs["kanonym_col"]].notnull()]
            k_pids=list(anon_desc["patientid"].unique())
            uniq_pids=list(set(set(k_pids).intersection(set(uniq_pids))))

        print("Number of included PIDs: {}".format(len(uniq_pids)))
        write_list_to_file(configs["inc_pid_list"], uniq_pids)

    # Respiratory failure, external validation
    elif configs["endpoint"]=="resp_extval":
        uniq_pids=[]
        for idx,fpath in enumerate(glob.glob(os.path.join(configs["merged_path"], "merged_*.parquet"))):
            df=pd.read_parquet(fpath)
            print("Index: {}".format(idx))
            uniq_pids.extend(list(df.admissionid.unique()))
            if configs["debug_mode"]:
                break
        print("PIDs in merged: {}".format(len(set(uniq_pids))))
        write_list_to_file(configs["inc_pid_list"], uniq_pids)

    # Wrong endpoint choice
    else:
        assert False

    
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
    parser.add_argument("--debug_mode", default=None, action="store_true", help="Only 1 batch")
    
    parser.add_argument("--gin_config", default="./configs/include_patients_extval.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/include_patients.gin", help="GIN config to use")
    
    configs=vars(parser.parse_args())
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    execute(configs)
