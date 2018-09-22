import json
import os
import numpy as np
import pandas as pd


DATA_FOLDER = "produced_data"
ROBUSTNESS_FOLDER = "robustness"
METRICS = ["ICC_A1"]#, "pearsonr", "pearsonr_pvalue"]


class Record(object):
    def __init__(self, split_id, smt_label, best_val_ep):
        self.split_id = split_id
        self.smt_label = smt_label
        self.best_val_ep = best_val_ep
        self.collect_test_aggregated_robustness()

    def collect_test_aggregated_robustness(self):
        path = os.path.join(
            DATA_FOLDER,
            self.smt_label,
            "{}_test_{}".format(ROBUSTNESS_FOLDER, self.best_val_ep),
            "robustness_measures",
            "feature_aggregation.json"
        )

        with open(path, 'r') as f:
            dic = json.load(f)

        self.value_table = {}
        self.split_stats = {}
        for pair_type, metric_dics in dic.items():
            if not pair_type.startswith("same"):
                continue
            for metric, values in metric_dics.items():
                if metric not in METRICS:
                    continue
                
                ss = ""
                for i, c in enumerate(pair_type):
                    if i == 0 or (i > 0 and pair_type[i - 1] == "_"):
                        ss += c
                score_id = "||".join([ss, metric])
                score_value = values["mean"]
                # for across runs stats
                self.value_table[score_id] = score_value

                # per run stat
                self.split_stats[score_id + "_" + "mean"] = values["mean"]
                self.split_stats[score_id + "_" + "std"] = values["std"]


def per_run_table(records):
    # sort records by split id
    records = sorted(records, key=lambda x: x.split_id)

    per_run = []
    keys = records[0].split_stats.keys()
    print(keys)
    for i, r in enumerate(records):
        row = [i]
        for k in keys:
            row.append(r.split_stats[k])
        per_run.append(row)

    header = ["epoch"] + list(keys)
    df = pd.DataFrame(
        data=np.array(per_run),
        columns=header
    )

    df = df.round(4)
    print(df.to_latex(index=False))
    
    
def summary_table(records):
    records = sorted(records, key=lambda x: x.split_id)
    
    keys = records[0].value_table.keys()
    # collect
    means = []
    for k in keys:
        vals = []
        for r in records:
            vals.append(r.value_table[k])
        means.append(np.mean(vals))
                     
    df = pd.DataFrame(
        data=np.array([means]),
        columns=list(keys)
    )
    
    df = df.round(4)
    print(df.to_latex(index=False))
