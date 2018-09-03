import os
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from db_utils.models import RecordGroup


from db_utils.db_connection import SumatraDB
import db_utils.compare_config as config


TABLE_NAME = config.TABLE_NAME
DB_PATH = config.DB_PATH
COLUMNS = config.COLUMNS
METRIC = config.METRIC
DATA_PATH = config.DATA_PATH
PLOT_TAG_LABEL = config.PLOT_TAG_LABEL
RECORD_LABEL = config.RECORD_LABEL
AGG_METRICS = config.AGG_METRICS
X_LABEL = config.X_LABEL
Y_LABEL = config.Y_LABEL
ID_TAGS = config.ID_TAGS
REPORT_METRICS = config.REPORT_METRICS

FILTERS = config.FILTERS
LEGEND_LOC = config.LEGEND_LOC
REASON = config.REASON


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

    for t in ID_TAGS:
        if g.find_tag(t) != rec.find_tag(t):
            return False

    return True

    # return (rec.version == g.version) and (rec.config == g.config)


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
    labels = []
    for group in groups:
        r = group[0]
        label = ",".join([r.find_tag(x) for x in PLOT_TAG_LABEL])
        labels.append(label)

    groups = [RecordGroup(g, l, DATA_PATH) for g, l in zip(groups, labels)]

    for g in groups:
        g.plot_group(
            metric=METRIC,
            x_label=X_LABEL,
            y_label=Y_LABEL,
            legend_loc=LEGEND_LOC
        )

    RecordGroup.compare_groups(
        groups=groups,
        metric=METRIC,
        x_label=X_LABEL,
        y_label=Y_LABEL
    )

    for g in groups:
        g.print_run_accuracies(REPORT_METRICS)


def main():
    # TODO: data base filter query
    db = SumatraDB(db=DB_PATH)
    if REASON is not None:
        records = db.get_filtered_by_reason(COLUMNS, REASON)
    elif RECORD_LABEL is not None:
        records = db.get_filtered_by_label(COLUMNS, RECORD_LABEL)
    else:
        records = db.get_all_records(COLUMNS)
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
