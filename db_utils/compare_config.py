# Global configuration
# Name of sumatra table containing records
TABLE_NAME = "django_store_record"
# Path to sqlite file
DB_PATH = ".smt/records"
# Column names of record table we are interested in
COLUMNS = ["label", "reason", "timestamp", "tags", "parameters_id", "version"]
# Metrics we want to plot 
METRICS = []
AGG_METRICS = ["test_accuracy_test", "test_accuracy_retest"]
# Path to folder containing records
DATA_PATH = "data"
# Tags that should be used to label lines
PLOT_TAG_LABEL = ["hidden_lambdas", "hidden_layer_regularizers"]

RECORD_LABEL = "20180515"

X_LABEL = "i-th epoch"
Y_LABEL = "test-retest accuracy"

# Filters should be used to filter out the experiments we
# want to compare. Can use one filter per 'type' of experiment.
# Every filter is applied to all the records.
filter_1 = {
    "tags": {
        "train_size": set(["2000"]),  # set of allowed values
        "weight_regularizer": set(["l2"]),
        "lambda_w": set(["0.005"]),
    }
}

filter_2 = {
    "tags": {
        "train_size": set(["2000"]),
        "lambda_f": set(["0.01", "0.05", "0.1", "0.2"])
    }
}

filter_3 = {
    "tags": {
        "hidden_lambdas": set(["[0]"])
    }
}

# Collect all the filters we want to apply
FILTERS = [filter_3]
