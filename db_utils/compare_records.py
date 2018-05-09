import os
import matplotlib.pyplot as plt
import numpy as np


from db_utils.db_connection import SumatraDB
import db_utils.compare_config as config


TABLE_NAME = config.TABLE_NAME
DB_PATH = config.DB_PATH
COLUMNS = config.COLUMNS
METRICS = config.METRICS
DATA_PATH = config.DATA_PATH
PLOT_TAG_LABEL = config.PLOT_TAG_LABEL
RECORD_LABEL = config.RECORD_LABEL

FILTERS = config.FILTERS


def filter_record_by_tag(rec, f):
    if "tags" not in f:
        return True

    for t in f["tags"]:
        vals = f["tags"][t]
        if len(vals) == 0:
            if t not in rec.tags:
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
    # color = iter(plt.cm.jet(np.linspace(0, 1, len(groups))))
    color = iter(plt.cm.tab10(np.linspace(0, 1, 11)))
    plt.figure(figsize=(6,4))

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
            
            if m not in r.metrics:
                continue
            
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
    if RECORD_LABEL is None:
        records = db.get_all_records(COLUMNS)
    else:
        records = db.get_filtered_by_label(COLUMNS, RECORD_LABEL)
    # print([str(r) for r in records])

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
