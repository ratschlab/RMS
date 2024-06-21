''' Readiness to extubate/ventilation statistics'''

import os
import os.path
import sys
import glob
import random
import argparse
import ipdb

import numpy as np
import pandas as pd

def execute(configs):

    ep_files=glob.glob(os.path.join(configs["endpoint_path"],"temporal_1",
                                    "batch_*.h5"))
    print("Number of endpoint batches: {}".format(len(ep_files)))

    if configs["small_sample"]:
        random.shuffle(ep_files)
        ep_files=ep_files[:20]
        
    n_patients=0
    n_pat_ready_ext=0
    n_pat_ready_ext_true=0
    n_pat_vent=0
        
    for fidx,ep_f in enumerate(ep_files):
        print("Processing batch {}/{}".format(fidx+1,len(ep_files)))
        df_ep=pd.read_hdf(ep_f,mode='r')
        pids=list(df_ep["PatientID"].unique())
        for pid in pids:
            n_patients+=1
            df_pid=df_ep[df_ep.PatientID==pid]
            n_finite=df_pid["readiness_ext"].notnull().sum()
            vent_vals=df_pid["vent_period"].values
            rext_vals=df_pid["readiness_ext"].values
            n_vent_active=np.sum(vent_vals==1)
            n_ready_exp=np.sum(rext_vals==1)
            if n_finite>0:
                n_pat_ready_ext+=1
            if n_ready_exp>0:
                n_pat_ready_ext_true+=1
            if n_vent_active>0:
                n_pat_vent+=1

        print("Patients w/ ready ext (any): {}/{}".format(n_pat_ready_ext,n_patients))
        print("Patients w/ ready ext (ready): {}/{}".format(n_pat_ready_ext_true,n_patients))
        print("Patients w/ vent active: {}/{}".format(n_pat_vent,n_patients))

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--endpoint_path", default="../../data/endpoints/hirid2_endpoints",
                        help="Endpoint path")

    # Output paths

    # Arguments
    parser.add_argument("--small_sample", default=False, action="store_true",
                        help="Use small sample?")

    configs=vars(parser.parse_args())

    execute(configs)
