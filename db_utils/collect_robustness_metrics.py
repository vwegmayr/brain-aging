import json
import os
import numpy as np
import pandas as pd

from robustness_records import RECORDS


DATA_FOLDER = "produced_data"
ROBUSTNESS_FOLDER = "robustness"


class Record(object):
    def __init__(self, split_id, smt_label, best_val_ep):
        self.split_id = split_id
        self.smt_label = smt_label
        self.best_val_ep = best_val_ep

    def collect_test_aggregated_robustness(self):
        path = os.path.join(
            DATA_FOLDER,
            "{}_test_{}".format(ROBUSTNESS_FOLDER, self.best_val_ep),
            "robustness_measures",
            "feature_aggregation.json"
        )

        with open(path, 'r') as f:
            dic = json.load(f)

        value_table = {}
        split_stats = {}
        for pair_type, metric_dics in dic.items():
            for metric, values in metric_dics.items():
                score_id = "||".join([pair_type, metric])
                score_value = values["mean"]
                # for across runs stats
                value_table[score_id] = score_value

                # per run stat
                split_stats[score_id + "_" + "mean"] = values["mean"]
                split_stats[score_id + "_" + "std"] = values["std"]


def per_run_table(records):
    # sort records by split id
    records = sorted(records, key=lambda x: x.split_id)

    per_run = []
    keys = records[0].split_stats.keys()

    for i, r in enumerate(records):
        row = [i]
        for k in keys:
            row.append(r.split_stats[k])
        per_run.append(row)

    header = ["epoch"] + list(keys)
    df = pd.DataFrame(
        data=np.array(per_run),
        header=header
    )

    df = df.round(4)
    print(df.to_latex(index=False))


if __name__ == "__main__":
    per_run_table(RECORDS)
