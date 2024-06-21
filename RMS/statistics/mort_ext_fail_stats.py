''' Determine the association of extubation failure with mortality
    with ventilated patients'''

import os
import os.path
import argparse
import glob
import ipdb

import pandas as pd
import numpy as np

def execute(configs):
    all_eps=glob.glob(os.path.join(configs["endpoint_path"],"temporal_1","batch_*.h5"))
    print("Number of endpoint files: {}".format(len(all_eps)))
    df_static=pd.read_hdf(configs["hirid_static_data"],mode='r')

    cnt_ext_fail_mort=0
    cnt_ext_fail_alive=0
    cnt_ext_no_fail_mort=0
    cnt_ext_no_fail_alive=0
    
    for ep_idx,ep_f in enumerate(all_eps):
        print("Processing batch: {}/{}".format(ep_idx+1,len(all_eps)))
        df_ep=pd.read_hdf(ep_f,mode='r')
        unique_pids=list(df_ep.PatientID.unique())
        for pid in unique_pids:
            df_pid=df_ep[df_ep.PatientID==pid]
            n_ext=df_pid.ext_failure.notnull().sum()
            if n_ext==0:
                continue

            df_static_pid=df_static[df_static.PatientID==pid]
            mort_status=df_static_pid["Discharge"].iloc[0]
            
            ext_fail_vals=df_pid.ext_failure.values
            if np.sum(ext_fail_vals==1)>0:
                if mort_status==4.0:
                    cnt_ext_fail_mort+=1
                else:
                    cnt_ext_fail_alive+=1

            else:
                if mort_status==4.0:
                    cnt_ext_no_fail_mort+=1
                else:
                    cnt_ext_no_fail_alive+=1

    ipdb.set_trace()
    print("FOO")

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--hirid_static_data", default="../../data/raw_data/hirid2/static.h5",
                        help="Path to HiRID static data")

    parser.add_argument("--hirid_label_dir", default="../../data/labels/hirid2_labels",
                        help="UMC labels")

    parser.add_argument("--endpoint_path", default="../../data/endpoints/hirid2_endpoints",
                        help="Endpoint path")    

    # Output paths

    # Arguments

    configs=vars(parser.parse_args())

    execute(configs)
