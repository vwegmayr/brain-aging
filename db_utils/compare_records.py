# Script can be used to compare numeric outcomes from the
# different experiments.
import matplotlib.pyplot as plt
import numpy as np


from db_utils.db_connection import SumatraDB


# Global configuration
# Name of sumatra table containing records
TABLE_NAME = "django_store_record"
# Path to sqlite file
DB_PATH = ".smt/records"
# Column names of record table we are interested in
COLUMNS = ["label", "reason", "timestamp", "tags", "parameters_id", "version"]
# Metrics we want to plot 
METRICS = ["test_accuracy_test"]
# Path to folder containing records
DATA_PATH = "data"
# Tags that should be used to label lines
PLOT_TAG_LABEL = ["lambda_w", "lambda_f"]

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


# Collect all the filters we want to apply
FILTERS = [filter_1, filter_2]


def filter_record_by_tag(rec, f):
    """
    Check if record satisfies filter requirements.

    Args:
        - rec: Record object
        - f: filter represented as dictonary (see top
          of file for example)

    Return:
        - True if record satisfies filter, False
          otherwise.
    """
    if "tags" not in f:
        return True

    for t in f["tags"]:
        vals = f["tags"][t]
        if (len(vals) == 0) and (t not in rec.tags):
            # no value necessary
            return False
        else:
            # multiple values for this tag allowed
            found = False
            for v in vals:
                tv = "{}={}".format(t, v)
                if tv in rec.tags:
                    found = True
                    break

            if not found:
                # print("tag {} not found in {}".format(tv, rec.tags))
                return False

    return True


def fits_group(rec, group):
    """
    Args:
        - rec: Record object
        - group: list of Record objects

    Return:
        - True if record matches group requirements,
          False otherwise.
    """
    g = group[0]

    return (rec.version == g.version) and (rec.config == g.config)


def group_records(records):
    """
    Find records that belong together, i.e. records
    that were executed with the same configuration
    and the same code version.

    Arg:
        - records: list of Record objects

    Return:
        - groups: list of list of records that have
          been grouped
    """
    groups = []  

    # Find group for each record
    for r in records:
        found = False
        for group in groups:
            if len(group) == 0:
                continue
            else:
                if fits_group(r, group):
                    found = True
                    group.append(r)

        # create new group
        if not found:
            groups.append([r])

    return groups


def plot_groups(groups):
    """
    Plot multiple groups on the same plot.

    Arg:
        - groups: list of grouped records
    """
    # generate color for each group
    # color = iter(plt.cm.jet(np.linspace(0, 1, len(groups))))
    color = iter(plt.cm.tab10(np.linspace(0, 1, 11)))

    for group in groups:
        for r in group:
            # Load metrics for records
            r.load_metrics(DATA_PATH)

    for m in METRICS:
        handles = []
        for group in groups:
            print("group of size {}".format(len(group)))
            print(",".join([g.label for g in group]))
            r = group[0]
            # Extract group label
            legend_label = ",".join([r.find_tag(x) for x in PLOT_TAG_LABEL])
            # x-label for plot
            x_label = r.metrics[m]["x_label"]
            # Only ticks for x values
            x = r.metrics[m]["x"]
            plt.xticks(x)

            # average y values
            ys = []
            for r in group:
                ys.append(r.metrics[m]["y"])

            ys = np.array(ys)
            std = np.std(ys, axis=0)
            mean = np.mean(ys, axis=0)
            plt.xlabel(x_label)
            plt.ylabel(m)
            c = next(color)
            line = plt.plot(x, mean, c=c, marker='o', linewidth=2,
                            label=legend_label)
            plt.plot(x, mean - std, c=c, linestyle="--", linewidth=0.5)
            plt.plot(x, mean + std, c=c, linestyle="--", linewidth=0.5)
            handles.append(line)

        plt.legend(loc=4, ncol=1)
        plt.grid(True)
        plt.show()


def main():
    # TODO: data base filter query
    db = SumatraDB(db=DB_PATH)
    records = db.get_all_records(COLUMNS)

    # Load config files
    for r in records:
        config = db.get_params_dic(r.params_id)
        r.config = config

    # Apply tag filters
    remaining = []
    for f in FILTERS:
        res = list(filter(lambda r: filter_record_by_tag(r, f), records))
        remaining.extend(res)

    print("{} records after tag filtering".format(len(remaining)))
    # Apply config filters

    # Groupy by
    groups = group_records(remaining)

    plot_groups(groups)


if __name__ == "__main__":
    main()
