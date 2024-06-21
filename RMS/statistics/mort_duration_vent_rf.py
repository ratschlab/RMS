''' Plots relating the fraction of stay in respiratory failure
    and mechanical ventilation to mortality in the ICU'''

import os
import os.path
import ipdb
import glob
import argparse
import random

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def execute(configs):
    df_static=pd.read_hdf(configs["hirid_static_data"],mode='r')

    endpoint_files=glob.glob(os.path.join(configs["endpoint_path"],"temporal_1","batch_*.h5"))
    random.shuffle(endpoint_files)
    mort_vent_ratio_map=dict()
    mort_rf_ratio_map=dict()
    
    for ep_idx,ep_f in enumerate(endpoint_files):
        print("Processing endpoint file: {}/{}".format(ep_idx+1,len(endpoint_files)))
        df_ep=pd.read_hdf(ep_f,mode='r')
        unique_pids=list(df_ep.PatientID.unique())
        for pid in unique_pids:
            df_pid=df_ep[df_ep.PatientID==pid]
            df_static_pid=df_static[df_static.PatientID==pid]
            mort_status=df_static_pid["Discharge"].iloc[0]
            mort_bin=1 if mort_status==4.0 else 0
            rf_status=df_pid["endpoint_status"].values
            vent_period=df_pid["vent_period"].values
            vent_ratio=100*np.sum(vent_period==1)/np.sum(np.isfinite(vent_period))
            for bin_left,bin_right in configs["RATIO_BINS"]:
                if vent_ratio>=bin_left and vent_ratio<bin_right:
                    bin_key=(bin_left,bin_right)
                    if bin_key not in mort_vent_ratio_map:
                        mort_vent_ratio_map[bin_key]=[]
                    mort_vent_ratio_map[bin_key].append(mort_bin)
                    break

            rf_ratio=100*np.sum((rf_status=="event_2") | (rf_status=="event_3"))/rf_status.size
            for bin_left,bin_right in configs["RATIO_BINS"]:
                if rf_ratio>=bin_left and vent_ratio<bin_right:
                    bin_key=(bin_left,bin_right)
                    if bin_key not in mort_rf_ratio_map:
                        mort_rf_ratio_map[bin_key]=[]
                    mort_rf_ratio_map[bin_key].append(mort_bin)
                    break                

    xs=[]
    ys=[]
    for k in sorted(mort_vent_ratio_map.keys()):
        arr=np.array(mort_vent_ratio_map[k])
        mort_ratio=100*np.sum(arr==1)/arr.size
        print("Vent ratio Bin: {}, Mort ratio: {:.2f}".format(k,mort_ratio))
        xs.append((k[1]+k[0])/2.)
        ys.append(mort_ratio)

    plt.plot(xs,ys,'go--',label="Mortality vs. vent ratio")
    plt.xlabel("Fraction of stay ventilated [%]")
    plt.ylabel("Discharge mortality [%]")
    plt.savefig(os.path.join(configs["plot_path"],"mort_vent_ratio.png"))
    plt.clf()

    xs=[]
    ys=[]
    for k in sorted(mort_rf_ratio_map.keys()):
        arr=np.array(mort_rf_ratio_map[k])
        mort_ratio=100*np.sum(arr==1)/arr.size
        print("RF ratio Bin: {}, Mort ratio: {:.2f}".format(k,mort_ratio))
        xs.append((k[1]+k[0])/2.)
        ys.append(mort_ratio)

    plt.plot(xs,ys,'go--',label="Mortality vs. RF")
    plt.xlabel("Fraction of stay in respiratory failure [%]")
    plt.ylabel("Discharge mortality [%]")    
    plt.savefig(os.path.join(configs["plot_path"],"mort_rf_ratio.png"))
    plt.clf()     
    

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--hirid_static_data", default="../../data/raw_data/hirid2/static.h5",
                        help="Path to HiRID static data")

    parser.add_argument("--endpoint_path", default="../../data/endpoints/hirid2_endpoints",
                        help="Endpoint path")        

    # Output paths
    parser.add_argument("--plot_path", default="../../data/plots/mort_duration",
                        help="Plotting path")            

    # Arguments

    configs=vars(parser.parse_args())

    configs["RATIO_BINS"]=[[0,10],[10,20],[20,30],[30,40],[40,50],
                           [50,60],[60,70],[70,80],[80,90],[90,100]]

    execute(configs)
