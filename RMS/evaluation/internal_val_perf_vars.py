''' Line plot which compactly summarizes Table 2'''

import os
import os.path
import sys
import argparse

import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({'figure.autolayout': True})

def execute(configs):
    x_vect=np.arange(1,21,1)
    y_vect_internal=configs["y_mean_internal"]
    disp_vect_internal=configs["y_disp_internal"]
    
    y_vect_val=configs["y_mean_val"]
    y_vect_diff=np.diff(y_vect_val)
    y_vect_diff=np.concatenate([np.array([y_vect_val[0]]),y_vect_diff])
    
    disp_vect_val=configs["y_disp_val"]

    fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True,figsize=(5,5),height_ratios=(0.5,0.5))

    # First part of the plot, internal performance
    ax1.plot(x_vect,y_vect_internal,label="HiRID->HiRID")
    ax1.fill_between(x_vect,y_vect_internal-0.5*disp_vect_internal,
                     y_vect_internal+0.5*disp_vect_internal,alpha=0.25)
    
    ax1.set_xticks(x_vect)    
    ax1.set_xlabel("Top Vars included",fontsize=6)
    ax1.grid(axis='both', alpha=0.25,linewidth=0.5,linestyle='-')
    ax1.set_ylabel("Prec @ 80 % Recall",fontsize=6)
    ax1.set_ylim((20,50))

    # Second part of the plot, validation performance
    x_bars=x_vect
    x_neg_data=x_bars[y_vect_diff<0]
    x_pos_data=x_bars[y_vect_diff>=0]
    y_neg_data=y_vect_diff[y_vect_diff<0]
    y_pos_data=y_vect_diff[y_vect_diff>=0]

    ax2.bar(x_neg_data,y_neg_data,width=1,color='r')
    ax2.bar(x_pos_data,y_pos_data,width=1,color='g')
    ax2.set_xlabel("# Top Vars included",fontsize=6)
    ax2.set_ylabel("Prec @ 80 % Recall, Val. diff when including var",fontsize=6)
    ax2.set_xticks(x_vect,configs["lst_vars"],rotation=30, horizontalalignment='right',fontsize=6)

    plt.savefig(os.path.join(configs["plot_path"],"top_vars_{}_perf.png".format(configs["task_name"])),dpi=400)

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths

    # Output paths
    parser.add_argument("--plot_path", default="../../data/plots/top_vars",
                        help="Plotting path")

    # Arguments
    #parser.add_argument("--task_name", default="resp_fail",
    #                    help="Name of tasks for which to produce plot")
    parser.add_argument("--task_name", default="ext_fail",
                        help="Name of tasks for which to produce plot")    

    configs=vars(parser.parse_args())

    # Results for particular task (respiratory failure or extubation failure)
    
    configs["y_mean_internal"]=np.array([24.5,29.9,31.6,31.6,32.6,31.1,32.8,33.6,33.6,34.7,
                                         34.1,34.2,34.5,35.1,35.1,34.3,34.7,34.9,33.9,33.3])
    configs["y_disp_internal"]=np.array([0.7,0.9,0.8,0.8,1.2,1.3,1.0,1.7,1.7,1.6,
                                         0.5,1.3,2.0,1.6,1.1,1.7,1.5,0.6,1.7,1.6])
    configs["y_mean_val"]=np.array([23.3,15.3,12.1,13.1,13.0,12.8,14.2,14.0,14.0,13.5,
                                    13.9,13.9,13.1,13.5,12.7,12.3,13.1,13.6,13.4,13.5])
    configs["y_disp_val"]=np.array([4.4,2.4,0.5,1.4,0.8,0.5,1.1,1.6,1.6,1.2,
                                    1.4,1.1,0.8,1.2,1.1,0.6,1.4,0.8,1.1,1.1])

    # configs["y_mean_internal"]=np.array([29.7,37.6,39.0,46.5,46.6,47.1,46.9,47.2,47.1,47.0,
    #                                      47.3,47.5,48.0,47.3,48.4,48.6,48.4,48.4,48.7,48.5])
    # configs["y_disp_internal"]=np.array([0.6,0.5,0.2,0.3,0.3,0.4,0.4,0.4,0.3,0.5,
    #                                      0.3,0.3,0.5,0.5,0.2,0.4,0.2,0.3,0.6,0.6])
    # configs["y_mean_val"]=np.array([23.0,28.2,30.2,37.2,32.7,32.2,32.9,31.9,30.0,30.8,
    #                                 30.5,29.9,30.5,29.9,29.6,30.9,30.0,30.6,29.3,30.4])
    # configs["y_disp_val"]=np.array([0.2,0.9,1.2,0.6,1.3,1.1,0.9,1.5,1.3,1.1,
    #                                 1.4,1.0,1.5,1.9,1.2,0.7,0.9,0.6,1.0,0.6])

    # Respiratory failure
    #configs["lst_vars"] = ['FiO$_2$*', 'SpO$_2$*', 'PaO$_2$*', 'Supplemental FiO$_2$ (%)*', 'Loop diuretics', 
    #                       'Heparin', 'Ventilator peak pressure*', 'Supplemental oxygen', 'Propofol', 'PEEPs*', 
    #                       'Pressure support*', 'GCS Motor*', 'Ventilator respiratory rate*', 'Sex', 'Estimated FiO$_2$*', 
    #                       'GCS Verbal*', 'Dobutamine', 'Benzodiacepine*', 'MV(exp)', 'PaCO$_2$*']

    # Extubation failure
    configs["lst_vars"] = ["Pressure support*","Benzodiacepine","Norepinephrine",'FiO$_2$*',
                           "Propofol*","MV(spont servo)","Ventilator respiratory rate*",
                           "RRsp. m","Ventilation presence*","Supplemental oxygen*",
                           "Insulin (fast acting)*","Loop diuretics","Supplemental FiO$_2$ (%)",
                           "MV(exp)*","Ventilator peak pressure","Heparin","PaCO$_2$*",
                           "Emergency admission*","PEEPs*","Ventilator mode group"]
    
    execute(configs)
