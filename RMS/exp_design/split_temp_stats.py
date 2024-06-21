''' Temporal stats for the splits in terms of break-point'''

import os
import os.path
import sys
import pickle
import ipdb
import argparse
import glob

import numpy as np
import pandas as pd


def execute(configs):

    batch_fs=glob.glob(os.path.join(configs["imputed_data_path"],"temporal_1","batch_*.h5"))
    pid_adm_map=dict()
    for bix,batch_f in enumerate(batch_fs):
        df=pd.read_hdf(batch_f,mode='r')
        print("Load batch: {}/{}".format(bix+1,len(batch_fs)))
        pids=list(df["PatientID"].unique())
        for pid in pids:
            df_pid=df[df.PatientID==pid]
            adm_time=df_pid["AbsDatetime"].min()
            pid_adm_map[pid]=adm_time

        if configs["debug_mode"]:
            break
    
    # Load one temporal split, determine the minimal admission date for the test set of that split
    with open(configs["split_desc_path"],'rb') as fp:
        obj=pickle.load(fp)
        split_test=obj["temporal_1"]["test"]
        split_train=obj["temporal_1"]["train"]
        split_val=obj["temporal_1"]["val"]

        valid_split_test=list(filter(lambda pid: pid in pid_adm_map, split_test))
        valid_split_val=list(filter(lambda pid: pid in pid_adm_map, split_val))
        valid_split_train=list(filter(lambda pid: pid in pid_adm_map, split_train))

        print("Number in valid train/val/test: {}/{}/{}".format(len(valid_split_train),
                                                                len(valid_split_val),
                                                                len(valid_split_test)))

        min_test=min([pid_adm_map[pid] for pid in valid_split_test])
        max_test=max([pid_adm_map[pid] for pid in valid_split_test])

        min_train=min([pid_adm_map[pid] for pid in valid_split_train])
        max_train=max([pid_adm_map[pid] for pid in valid_split_train])

        min_val=min([pid_adm_map[pid] for pid in valid_split_val])
        max_val=max([pid_adm_map[pid] for pid in valid_split_val])


if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--imputed_data_path", default="../../data/imputed/noimpute_hirid2/reduced", help="Imputed data path")

    parser.add_argument("--split_desc_path", default="../../data/exp_design/temp_splits_hirid2.pickle", help="Split descriptor path")

    # Output paths

    # Arguments
    parser.add_argument("--debug_mode", default=False, action="store_true", help="Debug mode")
    
    configs=vars(parser.parse_args())


    execute(configs)
