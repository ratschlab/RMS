'''
Cluster dispatcher script for the main batched imputation 
script
'''

import subprocess
import os
import os.path
import sys
import argparse
import gin

from RMS.utils.filesystem import delete_if_exist, create_dir_if_not_exist


def execute(configs):
    job_index=0
    mem_in_mbytes=configs["mem_in_mbytes"]
    n_cpu_cores=configs["n_cpu_cores"]
    n_compute_hours=configs["n_compute_hours"]
    compute_script_path=configs["compute_script_path"]
    bad_hosts=["le-g3-003","le-g3-007"]
    dset_base_dir_reduced=configs["bern_imputed_reduced_dir"]

    for split_key in configs["split_configs"]:

        base_dir=os.path.join(dset_base_dir_reduced,split_key)
        batch_range=list(range(configs["max_batch_idx"]))

        if not configs["dry_run"]:
            create_dir_if_not_exist(base_dir)

        for batch_idx in batch_range:

            if batch_idx in configs["skip_batches"]:
                print("Skipping batch: {}".format(batch_idx))
                continue

            if not configs["dry_run"]:
                delete_if_exist(os.path.join(base_dir,"batch_{}.h5".format(batch_idx)))

            print("Impute patient data for split {}".format(split_key))
            job_name="imputation_{}_{}".format(split_key,batch_idx)
            log_stdout_file=os.path.join(configs["log_dir"], "{}.stdout".format(job_name))
            log_stderr_file=os.path.join(configs["log_dir"], "{}.stderr".format(job_name))
            delete_if_exist(log_stdout_file)
            delete_if_exist(log_stderr_file)
            
            cmd_line=" ".join(["sbatch", "--mem-per-cpu {}".format(mem_in_mbytes),
                               "-n", "{}".format(n_cpu_cores),
                               "--time","{}:00:00".format(n_compute_hours),
                               "--mail-type FAIL",
                               "--exclude compute-biomed-18",
                               "--job-name", "{}".format(job_name), "-o", log_stdout_file, "-e", log_stderr_file, "--wrap",
                               '\"python3', compute_script_path, "--run_mode CLUSTER",
                               "--split_key {}".format(split_key), "--batch_idx {}".format(batch_idx), "--gin_config {}".format(configs["gin_config_path"]),'\"'])
            assert(" rm " not in cmd_line)
            job_index+=1

            if configs["dry_run"]:
                print(cmd_line)
            else:
                subprocess.call([cmd_line], shell=True)

    print("Generated {} jobs".format(job_index))


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
    parser.add_argument("--dry_run", action="store_true", default=None, help="Should the dispatcher be dry-run?")
    
    #parser.add_argument("--gin_config", default="./configs/cluster_dynamic.gin", help="GIN config file to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_dynamic_extval_impute.gin", help="GIN config file to use")
    parser.add_argument("--gin_config", default="./configs/cluster_dynamic_extval_noimpute.gin", help="GIN config file to use")    
    
    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    configs["skip_batches"]=[]

    execute(configs)
