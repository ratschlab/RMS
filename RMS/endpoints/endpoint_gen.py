''' Script generating the endpoints'''

import argparse
import os
import os.path
import glob
import ipdb
import sys
import gin
import random

import numpy as np
import pandas as pd

from endpoint_resp import endpoint_gen_resp

def execute(configs):
    ''' Dispatch to the correct endpoint generation function'''
    random.seed(configs["random_state"])

    if configs["endpoint"]=="circ":
        endpoint_gen_circ(configs)
    else:
        endpoint_gen_resp(configs)

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

    parser.add_argument("--debug_mode", default=None, action="store_true", help="Only one batch + do not write to FS")
    parser.add_argument("--verbose", action="store_true", default=None, help="Verbose messages")
    parser.add_argument("--small_sample", action="store_true", default=None, help="Use a small sample of PIDs")
    parser.add_argument("--batch_idx", type=int, default=None, help="Batch index to process")
    parser.add_argument("--split", default=None, help="Split to process")
    parser.add_argument("--run_mode", default=None, help="Execution mode")
    parser.add_argument("--endpoint", default=None, help="Endpoint to process")
    parser.add_argument("--random_state", default=None, help="Random seed to use for endpoint replicates")

    #parser.add_argument("--gin_config", default="./configs/ep.gin", help="GIN config to use")    
    parser.add_argument("--gin_config", default="./configs/ep_extval.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/ep_circ_ref.gin", help="GIN config to use")

    configs=vars(parser.parse_args())
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)    
    
    configs["VAR_ID"]=configs["VAR_IDS"]
    split_key=configs["split"]
    batch_idx=configs["batch_idx"]
    rseed=configs["random_state"]

    execute(configs)

    print("SUCCESSFULLY COMPLETED...")
    
