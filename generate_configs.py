# CUDA_VISIBLE_DEVICES=3 python run_experiments.py
# Script to run MNIST test-retest experiments
from subprocess import call
import yaml
import os
import time

params_to_run = {
    #"params,data_params,train_size": [2000, 10000],
    #"params,params,lambda_w": [0], #[0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1],
    #"params,params,lambda_f": [0],
    #"params,input_fn_config,num_epochs": [30],
    #"params,params,lambda_f": [0.005, 0.01, 0.05, 0.1, 0.2, 0.4],
    #"params,params,lambda_o": [0, 1, 2, 5, 10],
    #"params,params,output_regularizer": ["js_divergence"],
    #"params,params,weight_regularizer": ["l2_sq"],
    #"params,params,hidden_features_regularizer": ["l2_sq", "l2"],
    #"params,params,hidden_dim": [100]
    #"params,params,hidden_layer_sizes": [],
    #"params,params,hidden_layer_regularizers": [],
    "params,params,hidden_lambda": [0, 1]
}

#base_config_path = "configs/mnist/two_level_test_retest.yaml"
#base_config_path = "configs/mnist/test_retest_logistic.yaml"
base_config_path = "configs/debug/pair_ae.yaml"
#base_config_path = "configs/mnist/mtl_test_retest.yaml"
base_config = yaml.load(open(base_config_path, "r"))

#reason = "Two level logistic regression, vary hidden dimension."
#reason = "Test-retest logistic regression with bias and weight regularization"
#reason = "Test-retest logistic regression with bias and no weight regularization"
#reason = "Two level logistic regression with output regularization."
#reason = "Test-retest logistic regression with large JS-divergence regularization."
#reason = "Analyze convergence of test-retest logistic baseline."
#reason = "Test-retest js-divergence, confusion matrix."
#reason = "MTL divergence task weight s 50."
reason = "ANN with multiple layers and hidden regularization."

special_tags = ["hidden_reg", "ann"]
#special_tags = ["no_reg", "with_bias", "logistic_baseline"]
#special_tags = ["confusion_matrix"]
#special_tags = ["MTL_linear_body", "js_task_weight=50"]

# Create neede folders
CONFIGS_DONE = 'configs_done'
CONFIGS_RUNNING = 'configs_running'
CONFIGS_TO_DO = 'configs_to_do'

if not os.path.exists(CONFIGS_TO_DO):
    os.makedirs(CONFIGS_TO_DO)

if not os.path.exists(CONFIGS_RUNNING):
    os.makedirs(CONFIGS_RUNNING)

if not os.path.exists(CONFIGS_DONE):
    os.makedirs(CONFIGS_DONE)


def process(i, keys, tags):
    if i >= len(keys):
        # run experiment
        config_id = str(time.time()).replace(".", "")
        with open("{}/{}_config.yaml".format(CONFIGS_TO_DO, config_id), "w") as f:
            yaml.dump(base_config, f)

        # dump tags
        with open("{}/{}_tags.txt".format(CONFIGS_TO_DO, config_id), "w") as f:
            for k in tags:
                f.write("{}={}\n".format(k, str(tags[k])))

            for t in special_tags:
                f.write("{}\n".format(t))

        with open("{}/{}_reason.txt".format(CONFIGS_TO_DO, config_id), "w") as f:
            f.write("{}\n".format(reason))

    else:
        key = keys[i]
        vals = params_to_run[key]
        for v in vals:
            # set value
            # ASSUME 3 levels
            ks = key.split(",")
            cur = base_config
            for k in ks[:-1]:
                cur = cur[k]
            cur[ks[-1]] = v
            if isinstance(v, list):
                tags[ks[-1]] = "||".join([str(a) for a in v])
            else:
                tags[ks[-1]] = v
            # recurse
            process(i + 1, keys, tags)


def main():
    keys = list(params_to_run.keys())
    process(0, keys, {})


if __name__ == "__main__":
    main()
