''' Generate a chunk file from existing data'''

import glob
import os
import os.path
import ipdb
import argparse
import pickle

import pandas as pd

def execute(configs):
    batch_reverse_map={}
    batch_list_map={}
    output_file = configs["chunk_file_out"]
    
    all_fs=sorted(glob.glob(os.path.join(configs["merged_path"], "merged_*.parquet")),
                  key=lambda fpath: int(fpath.split("/")[-1].split("_")[1]))
    
    for fidx,fpath in enumerate(all_fs):
        print("Processing batch {}".format(fidx))
        df_batch=pd.read_parquet(fpath)
        all_pids=list(df_batch.admissionid.unique())
        for pid in all_pids:
            batch_reverse_map[pid]=fidx
        batch_list_map[fidx]=all_pids

    pickle_obj = {"chunk_to_pids": batch_list_map, "pid_to_chunk": batch_reverse_map}

    with open(output_file, 'wb') as fp:
        pickle.dump(pickle_obj, fp)

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--merged_path", default="/cluster/work/grlab/clinical/umcdb/preprocessed/merged/2023-04-23_rm_drugoor",
                        help="Merged path to load")

    # Output paths
    parser.add_argument("--chunk_file_out", help="Chunk file descriptor to save",
                        default="../../data/exp_design/umcdb_chunking_ALL.pickle")

    # Arguments

    configs=vars(parser.parse_args())
    
    execute(configs)

    
