
import argparse
import glob
import os
import os.path
import ipdb
import random

import numpy as np
import pandas as pd

def execute(configs):

    stat_counts_ready_and_success=0
    stat_counts_nready_and_success=0
    stat_counts_ready_and_failure=0
    stat_counts_nready_and_failure=0

    nready_count=0
    overall_count=0

    ep_files=glob.glob(os.path.join(configs["ep_path"],"batch_*.h5"))

    if configs["small_sample"]:
        random.shuffle(ep_files)
        ep_files=ep_files[:10]

    for fix,fpath in enumerate(ep_files):
        print("Batch {}/{}".format(fix+1,len(ep_files)))
        df_ep=pd.read_hdf(fpath,mode='r')
        ext_success_arr=np.array(df_ep.ext_success)
        readiness_ext_arr=np.array(df_ep.readiness_ext)
        stat_counts_ready_and_success+=np.sum((ext_success_arr==1.0) & (readiness_ext_arr==1.0))
        stat_counts_nready_and_success+=np.sum((ext_success_arr==1.0) & (readiness_ext_arr==0.0))
        stat_counts_ready_and_failure+=np.sum((ext_success_arr==0.0) & (readiness_ext_arr==1.0))
        stat_counts_nready_and_failure+=np.sum((ext_success_arr==0.0) & (readiness_ext_arr==0.0))

        nready_count+=np.sum(readiness_ext_arr==1.0)
        overall_count+=np.sum(np.isfinite(readiness_ext_arr))

    retrieve_suc_extube=stat_counts_ready_and_success/(stat_counts_ready_and_success+stat_counts_nready_and_success)
    retrieve_fail_extube=stat_counts_ready_and_failure/(stat_counts_ready_and_failure+stat_counts_nready_and_failure)

    retrieve_all=(stat_counts_ready_and_success+stat_counts_ready_and_failure)/(stat_counts_ready_and_success+stat_counts_ready_and_failure+ \
                                                                                stat_counts_nready_and_success+stat_counts_nready_and_failure)

    mark_rate=nready_count/overall_count

    print("Retrieve successful: {:.3f}, Retrieve failure: {:.3f}".format(retrieve_suc_extube, retrieve_fail_extube))
    print("Ready rate: {:.3f}".format(mark_rate))
    print("Retrieve all rate {:.3f}".format(retrieve_all))

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--ep_path", help="Path of endpoints to be analyzed",
                        default="/cluster/work/grlab/clinical/hirid2/research/3c_endpoints_resp/endpoints_210112_ext_threshold_8/point_est")

    # Output paths

    # Arguments
    parser.add_argument("--small_sample", default=False, action="store_true")

    configs=vars(parser.parse_args())
    
    execute(configs)
    

    
    

    
    
