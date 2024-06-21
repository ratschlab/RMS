''' Prints year counts for the training, val, test
    set of a particular splits'''

import os
import os.path
import argparse
import ipdb

import pandas as pd

from RMS.utils.admissions import lookup_admission_category, lookup_admission_time
from RMS.utils.io import load_pickle

def execute(configs):
    splits=load_pickle(configs["split_file"])
    split=splits[configs["split_id"]]

    df_patient_full=pd.read_hdf(configs["gen_table_path"], mode='r')

    for an_set in ["train","val","test"]:
        print("Analyzing set: {}".format(an_set))
        set_pids=split[an_set]
        year_counts={}
        for pid in set_pids:
            adm_time=lookup_admission_time(pid,df_patient_full)
            year=int(str(adm_time)[:4])
            if year not in year_counts:
                year_counts[year]=0
            year_counts[year]+=1
        print("Year counts: {}".format(year_counts))

    
    ipdb.set_trace()
    print("FOO")

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--split_file",
                        default="/cluster/work/grlab/clinical/hirid2/research/misc_derived/RESP_project/temp_splits_220627_kanonym.pickle",
                        help="Split file to use")

    #parser.add_argument("--split_file",
    #                    default="/cluster/work/grlab/clinical/hirid2/research/misc_derived/RESP_project/temp_splits_201103.pickle",
    #                    help="Split file to use")    

    parser.add_argument("--gen_table_path",
                        default="/cluster/work/grlab/clinical/hirid2/research/1a_hdf5_clean/v8/static.h5",
                        help="General patient table")

    # Output paths

    # Arguments
    parser.add_argument("--split_id", default="held_out",
                        help="Exact split to analyze")

    configs=vars(parser.parse_args())

    execute(configs)
