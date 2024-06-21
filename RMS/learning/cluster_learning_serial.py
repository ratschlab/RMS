''' 
Cluster dispatch script for ML learning
'''

import subprocess
import os
import os.path
import sys
import argparse
import itertools
import random
import ipdb
import glob
import csv
import json
import gin

from RMS.utils.filesystem import delete_if_exist, create_dir_if_not_exist


def execute(configs):
    random.seed(configs["random_seed"])
    job_index=0
    mem_in_mbytes=configs["mbytes_per_job"]
    n_cpu_cores=configs["num_cpu_cores"]
    n_compute_hours=configs["hours_per_job"]
    is_dry_run=configs["dry_run"]
    ml_model=configs["ml_model"]
    col_desc=configs["col_desc"]
    bad_hosts=configs["bad_hosts"]
    
    if not is_dry_run and not configs["preserve_logs"]:
        print("Deleting previous log files...")
        for logf in os.listdir(configs["log_dir"]):
            os.remove(os.path.join(configs["log_dir"], logf))

    for bern_split_key,umc_split_key in configs["SPLIT_CONFIGS"]:
        split_dir=os.path.join(configs["pred_dir"],"reduced",bern_split_key)

        if not is_dry_run:
            create_dir_if_not_exist(split_dir)

        for task_key,eval_key in configs["ALL_TASKS"]:
            output_base_key="{}_{}_{}".format(task_key, col_desc, ml_model)
            pred_output_dir=os.path.join(split_dir,output_base_key)
            if not is_dry_run:
                create_dir_if_not_exist(pred_output_dir)

            print("Fit ML model for split {}, task: {}, ML model: {}".format(bern_split_key,task_key,ml_model))
            job_name="mlfit_{}_{}_{}_{}".format(configs["col_desc"],bern_split_key,umc_split_key,task_key,ml_model)
            log_stdout_file=os.path.join(configs["log_dir"],"{}.stdout".format(job_name))
            log_stderr_file=os.path.join(configs["log_dir"],"{}.stderr".format(job_name))            
            delete_if_exist(log_stdout_file)
            delete_if_exist(log_stderr_file)            

            if ml_model in ["lightgbm","tree","logreg"]:
                cmd_line=" ".join(["sbatch", "--mem-per-cpu {}".format(mem_in_mbytes), 
                                   "-n", "{}".format(n_cpu_cores),
                                   "--time", "{}:00:00".format(n_compute_hours),
                                   "--exclude=compute-biomed-10,compute-biomed-21",
                                   "--mail-type FAIL",
                                   "--partition=compute",
                                   "--job-name","{}".format(job_name), "-o", log_stdout_file, "-e", log_stderr_file, "--wrap",
                                   '\"python3', configs["compute_script_path"], "--run_mode CLUSTER", "--gin_config {}".format(configs["script_gin_file"]),
                                   "--bern_split_key {}".format(bern_split_key),
                                   "--umc_split_key {}".format(umc_split_key),                                   
                                   "--column_set {}".format(configs["col_desc"]),
                                   "--label_key {}".format(task_key),"" if eval_key is None else "--eval_label_key {}".format(eval_key),
                                   "--ml_model {}".format(ml_model), '\"'])

            assert(" rm " not in cmd_line)
            job_index+=1

            if configs["dry_run"]:
                print("CMD: {}".format(cmd_line))
            else:
                subprocess.call([cmd_line], shell=True)

                if configs["debug_mode"]:
                    sys.exit(0)

    print("Generated {} jobs...".format(job_index))


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

    parser.add_argument("--dry_run", default=None, action="store_true", help="Should a dry-run be used?")
    parser.add_argument("--debug_mode", default=None, action="store_true", help="Debugging mode, run only one job")
    parser.add_argument("--preserve_logs", default=None, action="store_true", help="Should logging files be preserved?")
    parser.add_argument("--1percent_sample", default=None, action="store_true", help="Should a 1 % sample of train/val be used, for debugging")
    
    #parser.add_argument("--gin_config", default="./configs/cluster_internal_rf.gin", help="GIN config to use") 
    #parser.add_argument("--gin_config", default="./configs/cluster_internal_rf_short_horizon.gin", help="GIN config to use")   
    #parser.add_argument("--gin_config", default="./configs/cluster_internal_rf_prefix.gin", help="GIN config to use")

    #parser.add_argument("--gin_config", default="./configs/cluster_rf_50pct.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_rf_25pct.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_rf_10pct.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_rf_5pct.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_rf_2pct.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_rf_1pct.gin", help="GIN config to use")
    
    #parser.add_argument("--gin_config", default="./configs/cluster_internal_ef.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_internal_eflite.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_internal_ef_prefix.gin", help="GIN config to use")    

    #parser.add_argument("--gin_config", default="./configs/cluster_ef_50pct.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_ef_25pct.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_ef_10pct.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_ef_5pct.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_ef_2pct.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_ef_1pct.gin", help="GIN config to use")
    
    #parser.add_argument("--gin_config", default="./configs/cluster_internal_rexp.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_internal_vent.gin", help="GIN config to use")        

    #parser.add_argument("--gin_config", default="./configs/cluster_umcdb_val_rf.gin", help="GIN config to use")
    parser.add_argument("--gin_config", default="./configs/cluster_umcdb_transport_val_rf.gin", help="GIN config to use")    
    #parser.add_argument("--gin_config", default="./configs/cluster_umcdb_val_ef.gin", help="GIN config to use")    

    #parser.add_argument("--gin_config", default="./configs/cluster_umcdb_retrain_rf.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_umcdb_retrain_rfp.gin", help="GIN config to use")    
    #parser.add_argument("--gin_config", default="./configs/cluster_umcdb_retrain_ef.gin", help="GIN config to use") 
    #parser.add_argument("--gin_config", default="./configs/cluster_umcdb_retrain_eflite.gin", help="GIN config to use")           
    
    #parser.add_argument("--gin_config", default="./configs/cluster_internal_no_pharma.gin", help="GIN config to use")  
    #parser.add_argument("--gin_config", default="./configs/cluster_umcdb_val_no_pharma.gin", help="GIN config to use")   
    #parser.add_argument("--gin_config", default="./configs/cluster_umcdb_retrain_no_pharma.gin", help="GIN config to use")     

    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    execute(configs)
