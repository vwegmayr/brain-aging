import os
import matplotlib.pyplot as plt
import numpy as np


from db_utils.db_connection import SumatraDB


TABLE_NAME = "django_store_record"
COLUMNS = ["label", "reason", "timestamp", "tags", "parameters_id", "version"] 
METRICS = ["test_accuracy_test"]
DATA_PATH = "data"


filter_1 = {
    "tags": {
        "train_size": set(["500"]),  # set of allowed values
    },
    "config": {
        "lambda_f": [0.01, 0.05, 0.1, 0.2]
    }
}

filter_2 = {
    "tags": {
        "train_size": set(["500"]),
        "baseline": set([]),
    }
}


FILTERS = [filter_1]


def filter_record_by_tag(rec, f):
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
                return False

    return True


def fits_group(rec, group):
    g = group[0]

    return (rec.version == g.version) and (rec.config == g.config)


def group_records(records):
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
    # generate color for each group
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(groups))))

    for group in groups:
        for r in group:
            # Load metrics for records
            r.load_metrics(DATA_PATH)

    for m in METRICS:
        for group in groups:
            print("group of size {}".format(len(group)))
            print(",".join([g.label for g in group]))
            r = group[0]
            x_label = r.metrics[m]["x_label"]
            x = r.metrics[m]["x"]
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
            plt.plot(x, mean, c=c)
            plt.plot(x, mean - std, c=c)
            plt.plot(x, mean + std, c=c)

        plt.show()


def main():
    # TODO: data base filter query
    db = SumatraDB()
    records = db.get_all_records(COLUMNS)
    # Load config files
    for r in records:
        config = db.get_params_dic(r.params_id)
        r.config = config

    # Apply tag filters
    for f in FILTERS:
        records = list(filter(lambda r: filter_record_by_tag(r, f), records))

    # Apply config filters

    # Groupy by
    groups = group_records(records)

    plot_groups(groups)


if __name__ == "__main__":
    main()
