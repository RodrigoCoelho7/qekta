import argparse
import yaml
from collections import namedtuple
import ray
import os
import datetime
import shutil
from ray import tune
from utils.config_utils import create_config
from training_functions.run import run
from training_functions.random_initialization import random_init
from training_functions.gradient import get_gradient


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "This parser receives the yaml config file")
    parser.add_argument("--config", default = "configs/test.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())
    path = None

    ray.init(local_mode = config.ray_local_mode,
             num_cpus = config.num_cpus,
             num_gpus=config.num_gpus,
             _temp_dir=os.path.dirname(os.getcwd()) + '/ray_logs',
             include_dashboard = False)
    
    param_space = create_config(config)
    
    name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config.method + '_'
    ray_path = os.getcwd() + '/' + config.ray_logging_path
    path = ray_path + "/" + name

    os.makedirs(os.path.dirname(path + '/'), exist_ok=True)
    shutil.copy(args.config, path + '/alg_config.yml')

    def trial_name_creator(trial):
            return trial.__str__() + '_' + trial.experiment_tag + ','
    
    if config.type == "random_init":
        trainable = tune.with_resources(random_init, {"cpu": config.num_cpus_worker})
    elif config.type == "train":
        trainable = tune.with_resources(run, {"cpu": config.num_cpus_worker})
    elif config.type == "gradient":
        trainable = tune.with_resources(get_gradient, {"cpu": config.num_cpus_worker})

    tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(num_samples=config.ray_num_trial_samples,
                                        trial_dirname_creator=trial_name_creator),
            param_space=param_space)
        
    tuner.fit()
    ray.shutdown()