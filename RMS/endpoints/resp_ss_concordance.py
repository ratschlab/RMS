''' Script to compare the concordance of endpoints with different subsampling settings, taking
    into account different random replicates'''

import glob
import os
import os.path
import argparse
import random
import ipdb

import pandas as pd
import numpy as np

def execute(configs):

    rsplits=list(range(10))
    event_levels=["event_3","event_2","event_1","event_0"]
    out_dict={"ep_level": [], "subsample_config": [], "temporal_coherence": []}

    for ep_level in event_levels:

        for trace_label, trace_path in [("ss_both", configs["ss_both_path"])]:

            print("Analyzing endpoint level: {}, with config: {}".format(ep_level,trace_label))
            batch_fs=sorted(glob.glob(os.path.join(trace_path,"seed_0", "batch_*.h5")))

            if configs["subsample"] is not None:
                random.seed(configs["random_state"])                
                batch_fs=random.sample(batch_fs,configs["subsample"])

            coherence_cnt=0
            all_cnt=0

            for bidx,fpath in enumerate(batch_fs):
                batch_id=fpath.split("/")[-1].split("_")[1][:-3]
                print("Processing batch: {}/{}: {}".format(bidx+1,len(batch_fs),batch_id))
                ref_df=pd.read_hdf(os.path.join(configs["full_path"],"batch_{}.h5".format(batch_id)))
                adapt_df={}
                for rseed in rsplits:
                    fpath=os.path.join(trace_path,"seed_{}".format(rseed),"batch_{}.h5".format(batch_id))
                    adapt_df[rseed]=pd.read_hdf(fpath,mode='r')

                pids=ref_df.PatientID.unique()
                for pix,pid in enumerate(pids):
                    if configs["verbose"] and (pix+1)%10==0:
                        print("Processing PID {}/{}".format(pix+1,len(pids)))

                    adapt_pid_df={}
                    for k in adapt_df.keys():
                        df=adapt_df[k]
                        adapt_pid_df[k]=df[df.PatientID==pid]
                    df_pid_ref=ref_df[ref_df.PatientID==pid]
                    for k in adapt_pid_df.keys():
                        assert(adapt_pid_df[k].shape[0]==df_pid_ref.shape[0])

                    df_ref_event=df_pid_ref[df_pid_ref["endpoint_status"]==ep_level]

                    event_arr=np.array(df_pid_ref["endpoint_status"]==ep_level)
                        
                    and_rows=[]

                    # Iterate over all random seed arrays
                    for k in adapt_pid_df.keys():
                        df_pid=adapt_pid_df[k]
                        df_mod_pid=df_pid[df_pid_ref["endpoint_status"]==ep_level]
                        and_rows.append(np.array(df_ref_event["endpoint_status"]==df_mod_pid["endpoint_status"]))

                    and_arr=np.vstack(and_rows)
                    coh_cnt=np.all(and_arr,axis=0).sum()
                    all_cnt+=df_ref_event.shape[0]
                    coherence_cnt+=coh_cnt

            print("Temporal coherence of label {} in {}: {:.3f}".format(trace_label, ep_level, coherence_cnt/all_cnt))
            out_dict["ep_level"].append(ep_level)
            out_dict["subsample_config"].append(trace_label)
            out_dict["temporal_coherence"].append(coherence_cnt/all_cnt)

    out_df=pd.DataFrame(out_dict)
    out_df.to_csv(os.path.join(configs["table_path"], "coherence_analysis.tsv"),sep='\t',index=False)

if __name__=="__main__":

    parser=argparse.ArgumentParser()
    
    # Input paths
    parser.add_argument("--full_path",
                        default="/cluster/work/grlab/clinical/hirid2/research/3c_endpoints_resp/endpoints_210112/point_est",
                        help="Path of the full endpoint as ref")
    
    parser.add_argument("--ss_spo2_path",
                        default="/cluster/work/grlab/clinical/hirid2/research/3c_endpoints_resp/endpoints_210112_ss_spo2/point_est",
                        help="Path of the spo2 SS endpoint")
    
    parser.add_argument("--ss_abga_path",
                        default="/cluster/work/grlab/clinical/hirid2/research/3c_endpoints_resp/endpoints_210112_ss_abga/point_est",
                        help="Path of the abga SS endpoint")
    
    parser.add_argument("--ss_both_path",
                        default="/cluster/work/grlab/clinical/hirid2/research/3c_endpoints_resp/endpoints_210112_ss_abga_spo2/point_est",
                        help="Path of the both SS endpoint")

    # Output paths
    parser.add_argument("--table_path", default="/cluster/work/grlab/clinical/hirid2/research/10c_results_resp/endpoint_robustness",
                        help="Path where to store result tables")

    # Arguments
    parser.add_argument("--subsample", default=10,type=int, help="How many batch files to subsample?")
    parser.add_argument("--random_state", default=2021, type=int)
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose messages")

    configs=vars(parser.parse_args())

    execute(configs)
