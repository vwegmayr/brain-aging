import json
import os
import numpy as np
import pandas as pd
import yaml
import pprint


DATA_FOLDER = "produced_data"
ROBUSTNESS_FOLDER = "robustness"
METRICS = ["ICC_A1"]#, "pearsonr", "pearsonr_pvalue"]


class Record(object):
    def __init__(self, split_id, smt_label, best_val_ep):
        self.split_id = split_id
        self.smt_label = smt_label
        self.best_val_ep = best_val_ep
        self.collect_test_aggregated_robustness()

        # Read diag dim
        config_path = os.path.join(
            "data",
            smt_label,
            "config.yaml"
        )

        with open(config_path, 'r') as f:
            config = yaml.load(f)

        self.diag_dim = config["params"]["params"]["diagnose_dim"]
        self.hidden_dim = config["params"]["params"]["hidden_dim"]

    def is_regularized(self, feature):
        feature = int(feature)
        if feature >= self.hidden_dim - self.diag_dim:
            return True
        else:
            return False

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

    def reg_vs_not_reg_epoch(self, epoch):
        path = os.path.join(
            DATA_FOLDER,
            self.smt_label,
            "{}_test_{}".format(ROBUSTNESS_FOLDER, self.best_val_ep),
            "robustness_measures"
        )

        all_summs = {}
        for fname in os.listdir(path):
            # check if file contains per feature computations
            if fname.endswith('computations.json'):
                summ = self.reg_vs_not_reg_file(os.path.join(path, fname))

                for reg in summ.keys():
                    for m in summ[reg].keys():
                        values = summ[reg][m]
                        summ[reg][m] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "median": np.median(values)
                        }

                k = "_".join(fname.split("_")[:-1])
                all_summs[k] = summ

        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(all_summs) 

    def reg_vs_not_reg_file(self, file_path):
        with open(file_path, 'r') as f:
            dic = json.load(f)

        features = list(dic.keys())
        summ = {
            "reg": {},
            "not_reg": {}
        }
        for f in features:
            metric_dic = dic[f]
            if self.is_regularized(f):
                summ_dic = summ["reg"]
            else:
                summ_dic = summ["not_reg"]

            for m, val in metric_dic.items():
                if m not in summ_dic:
                    summ_dic[m] = []
                summ_dic[m].append(val)
        
        return summ

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
    means = ["mean"]
    medians = ["median"]
    for k in keys:
        vals = []
        for r in records:
            vals.append(r.value_table[k])
        means.append(np.mean(vals))
        medians.append(np.median(vals))
                     
    df = pd.DataFrame(
        data=np.array([means, medians]),
        columns=["agg"] + list(keys)
    )
    
    df = df.round(4)
    print(df.to_latex(index=False))


def reg_vs_not_reg(records):
    records = sorted(records, key=lambda x: x.split_id)

    for r in records:
        r.reg_vs_not_reg_epoch(r.best_val_ep)    
