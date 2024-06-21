''' Per split statistics on the readiness to extubate 
    endpoint'''

import random
import glob
import os
import os.path
import ipdb
import argparse

import pandas as pd
import numpy as np

def lookup_admission_year(pid, df_patient_full):
    ''' Looks up a proxy to admission time for a PID'''
    df_patient=df_patient_full[df_patient_full["PatientID"]==pid]

    if not df_patient.shape[0]==1:
        return None
    adm_time=np.array(df_patient["AdmissionTime"])[0]
    return int(str(adm_time)[:4])

def execute(configs):
    random.seed(2021)
    df_out_dict={"VENT_MODE": [],
                 "PEEP": [],
                 "PSUPPORT": [],
                 "FIO2": [],
                 "SBIDX": [],
                 "RRATE": [],
                 "MINVOL": [],
                 "PFRATIO": [],
                 "PACO2": [],
                 "GCS": [],
                 "MAP": [],
                 "DRUGS": [],
                 "LACTATE": [],
                 "SPLIT": []}

    df_static=pd.read_hdf(configs["static_path"],mode='r')
    
    for split in configs["SPLITS"]:
        print("Statistics for split {}".format(split))
        df_out_dict["SPLIT"].append(split)

        pos_counts={}
        fin_counts={}

        all_eps=sorted(glob.glob(os.path.join(configs["endpoint_path"], split,"batch_*.h5")))

        if configs["small_sample"]:
            random.shuffle(all_eps)
            all_eps=all_eps[:20]

        for fix,fpath in enumerate(all_eps):
            if (fix+1)%10==0:
                print("File {}/{}".format(fix+1,len(all_eps)))
            df=pd.read_hdf(fpath,mode='r')

            for pid in list(df.PatientID.unique()):
                df_pid=df[df.PatientID==pid]
                adm_year=lookup_admission_year(pid, df_static)
                for col in configs["REL_COLS"]:
                    if (col,adm_year) not in pos_counts:
                        pos_counts[(col,adm_year)]=0
                        fin_counts[(col,adm_year)]=0
                    arr=np.array(df_pid[col])
                    pos_counts[(col,adm_year)]+=np.sum(arr==1)
                    fin_counts[(col,adm_year)]+=np.sum(np.isfinite(arr))

        year_support=[year for year in range(2000,2022) if ("ext_not_ready_vent_mode", year) in pos_counts.keys()]
                    
        prev={}
        for year in year_support:
            prev[year]=pos_counts[("ext_not_ready_vent_mode",year)]/fin_counts[("ext_not_ready_vent_mode",year)]

        ipdb.set_trace()
                    
        df_out_dict["VENT_MODE"].append(pos_counts["ext_not_ready_vent_mode"]/fin_counts["ext_not_ready_vent_mode"])
        df_out_dict["PEEP"].append(pos_counts["ext_not_ready_peep"]/fin_counts["ext_not_ready_peep"])
        df_out_dict["PSUPPORT"].append(pos_counts["ext_not_ready_psupport"]/fin_counts["ext_not_ready_psupport"])
        df_out_dict["FIO2"].append(pos_counts["ext_not_ready_fio2"]/fin_counts["ext_not_ready_fio2"])
        df_out_dict["SBIDX"].append(pos_counts["ext_not_ready_sbidx"]/fin_counts["ext_not_ready_sbidx"])
        df_out_dict["RRATE"].append(pos_counts["ext_not_ready_rr"]/fin_counts["ext_not_ready_rr"])
        df_out_dict["MINVOL"].append(pos_counts["ext_not_ready_minvol"]/fin_counts["ext_not_ready_minvol"])
        df_out_dict["PFRATIO"].append(pos_counts["ext_not_ready_pfratio"]/fin_counts["ext_not_ready_pfratio"])
        df_out_dict["PACO2"].append(pos_counts["ext_not_ready_paco2"]/fin_counts["ext_not_ready_paco2"])
        df_out_dict["GCS"].append(pos_counts["ext_not_ready_gcs"]/fin_counts["ext_not_ready_gcs"])
        df_out_dict["MAP"].append(pos_counts["ext_not_ready_map"]/fin_counts["ext_not_ready_map"])
        df_out_dict["DRUGS"].append(pos_counts["ext_not_ready_drugs"]/fin_counts["ext_not_ready_drugs"])
        df_out_dict["LACTATE"].append(pos_counts["ext_not_ready_lactate"]/fin_counts["ext_not_ready_lactate"])

    df_out=pd.DataFrame(df_out_dict)
    ipdb.set_trace()
    print("FOO")
                
if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--endpoint_path", default="/cluster/work/grlab/clinical/hirid2/research/3c_endpoints_resp/endpoints_210407",
                        help="Endpoint path")
    parser.add_argument("--static_path", default="/cluster/work/grlab/clinical/hirid2/research/1a_hdf5_clean/v8/static.h5")

    # Output paths

    # Arguments
    parser.add_argument("--small_sample", default=False, action="store_true")

    configs=vars(parser.parse_args())

    # configs["SPLITS"]=["point_est","temporal_1","temporal_2",
    #                    "temporal_3","temporal_4","temporal_5"]

    configs["SPLITS"]=["temporal_4"]

    configs["REL_COLS"]=['ext_not_ready_vent_mode',
                         'ext_not_ready_peep', 'ext_not_ready_psupport', 'ext_not_ready_fio2',
                         'ext_not_ready_sbidx', 'ext_not_ready_rr', 'ext_not_ready_minvol',
                         'ext_not_ready_pfratio', 'ext_not_ready_paco2', 'ext_not_ready_gcs',
                         'ext_not_ready_map', 'ext_not_ready_drugs', 'ext_not_ready_lactate']

    
    execute(configs)

                        
