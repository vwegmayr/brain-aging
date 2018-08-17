# Global configuration
# Name of sumatra table containing records
TABLE_NAME = "django_store_record"
# Path to sqlite file
DB_PATH = ".smt/records"
# Column names of record table we are interested in
COLUMNS = ["label", "reason", "timestamp", "tags", "parameters_id", "version"]
# Metrics we want to plot
METRIC = "test_acc"
#METRICS = ["train_prediction_robustness_healthy_LogisticRegression_not_balanced_train_predictions_equal_pairs"]
AGG_METRICS = ["test_acc"]
#AGG_METRICS = ["test_accuracy_test", "test_accuracy_retest"]
#AGG_METRICS = ["test_acc_TestRetestCrossEntropy_preds_test", "test_acc_TestRetestCrossEntropy_preds_retest"]
X_LABEL = "i-th epoch"
Y_LABEL = "Test accuracy"
#METRICS += ["test_acc_TestRetestCrossEntropy_preds_test"]
# Path to folder containing records
DATA_PATH = "data"
# Tags that should be used to label lines
PLOT_TAG_LABEL = ["hidden_lambda"]
ID_TAGS = ["hidden_lambda"]
#PLOT_TAG_LABEL = ["js_task_weight"]
LEGEND_LOC = 0

REASON = "Unet CV"
RECORD_LABEL = None

# Filters should be used to filter out the experiments we
# want to compare. Can use one filter per 'type' of experiment.
# Every filter is applied to all the records.
filter_1 = {
    "tags": {
        "train_size": set(["2000"]),  # set of allowed values
        #"weight_regularizer": set(["l2", "l2_sq"]),
        "lambda_f": set(["0", "0.4", "0.1"]),
        "hidden_features": set([])
    }
}

filter_2 = {
    "tags": {
        "train_size": set(["500"]),
        "lambda_o": set(["1", "1.5"]),
        "output_regularizer": set(["js_divergence"]),
        "confusion_matrix": set([])
    }
}


filter_3 = {
    "tags": {
        "train_size": set(["2000"]),
        "js_task_weight": set(["1", "10", "50"])
}
}

filter_4 = {
    "tags": {
        "train_size": set(["2000"]),
        "output_regularizer": set(["js_divergence"]),
        "ann": set([]),
        "lambda_o": set(["0", "1", "5", "10"])
}
}

# baseline
filter_5 = {
    "tags": {
        "no_reg": set([]),
        "with_bias": set([]),
        "logistic_baseline": set([]),
        "lambda_w": set(["0"]),
        "train_size": set(["10000"]),
    }
}

# baseline ann
filter_6 = {
    "tags": {
        "ann": set([]),
        "hidden_lambdas": set(["0||0||0"]),
        "train_size": set(["2000"]),
    }
}

filter_7 = {
    "tags": {
        #"ann": set([]),
        #"hidden_lambdas": set(["0"]),
        "train_size": set(["10000"]),
        "hidden_layer_regularizers": set(["None||None"]),
        "hidden_layer_sizes": set(["512||512"]),
        "mix_pairs": set(["True", "False"]),
    #"hidden_lambdas": set(["0.01||0", "0.1||0"])
    }
}

filter_8 = {
    "tags": {
        "MTL_non_linear": set([]),
        "train_size": set(["10000"]),
    }
}
    
filter_9 = {
    "tags": {
        "hidden_dim": set(["64"]),
        "diagnose_dim": set(["32"]),
        "n_folds": set(["5"]),
        "lin_ae": set([]),
        "test_fold": set(["0", "1", "2", "3", "4"])
    }
}
    
filter_10 = {
    "tags": {
        "hidden_dim": set(["64"]),
        "diagnose_dim": set(["32"]),
        "n_folds": set(["5"]),
        "seed": set(["13"]),
        "conv_ae": set([]),
        "test_fold": set(["0", "1", "2", "3", "4"]),
        "class": set(["src.data.streaming.mri_streaming.SimilarPairStream"]),
    }
}
    
filter_11 = {
    "tags": {
        "hidden_lambda": set(["0", "1"]),
    }
}

# Collect all the filters we want to apply
FILTERS = [filter_11]
