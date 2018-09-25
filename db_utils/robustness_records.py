import json
import os
import numpy as np
import pandas as pd
import yaml
import pprint


DATA_FOLDER = "produced_data"
ROBUSTNESS_FOLDER = "robustness"
METRICS = ["ICC_A1"]#, "pearsonr", "pearsonr_pvalue"]


def shorten_pair_type(pair_type):
    ss = ""
    for i, c in enumerate(pair_type):
        if i == 0 or (i > 0 and pair_type[i - 1] == "_"):
            ss += c
    return ss


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
        self.n_epochs = config["params"]["input_fn_config"]["num_epochs"]

        # load sumatra scores
        sum_path = os.path.join(
            "data",
            smt_label,
            "sumatra_outcome.json"
        )

        with open(sum_path) as f:
            self.sumatra_outcome = json.load(f)

    def get_sumatra_values(self, metric_name):
        return self.sumatra_outcome["numeric_outcome"][metric_name]["y"]

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

        # pp = pprint.PrettyPrinter(indent=2)
        # pp.pprint(all_summs)
        value_table = {}
        for pair_type in all_summs.keys():
            if not pair_type.startswith("same"):
                continue
            ss = shorten_pair_type(pair_type)
            for reg in all_summs[pair_type].keys():
                for m, m_dic in all_summs[pair_type][reg].items():
                    if m not in METRICS:
                        continue
                    for k, v in m_dic.items():
                        score_id = "||".join([ss, reg, m, k])
                        value_table[score_id] = v
        
        return value_table

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

    def find_best_agreement_epoch(self):
        agreements = []
        for i in range(self.n_epochs):
            path = os.path.join(
                DATA_FOLDER,
                self.smt_label,
                "{}_test_{}".format(ROBUSTNESS_FOLDER, i),
                "robustness_measures",
                "feature_aggregation.json"
            )

            with open(path, 'r') as f:
                dic = json.load(f)

            agreement_score = 0.5 * dic["same_patient_healthy__healthy"]["ICC_A1"]["median"] +\
                0.5 * dic["same_patient_health_ad__health_ad"]["ICC_A1"]["median"]

            agreements.append(agreement_score)

        best_ep = np.argmax(agreements)
        # get test accuracy
        test_accs = self.get_sumatra_values("test_acc")

        print("best agreement of {} in epoch {} with test acc {}".format(
            agreements[best_ep], best_ep, test_accs[best_ep]
        ))
        
        return best_ep, agreements[best_ep], test_accs[best_ep]


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
    print(">>>>> reg vs no reg")
    records = sorted(records, key=lambda x: x.split_id)

    values = {}
    dics = []
    for r in records:
        dic = r.reg_vs_not_reg_epoch(r.best_val_ep)
        dics.append(dic)

    def print_table(regb=True):
        keys = dics[0].keys()
        for dic in dics:
            for k in keys:
                if k not in values:
                    values[k] = []
                values[k].append(dic[k])

        columns = []
        used_keys = []
        for k in keys:
            if regb:
                if not ("not" in k):
                    columns.append(values[k])
                    used_keys.append(k)
            else:
                if "not" in k:
                    columns.append(values[k])
                    used_keys.append(k)

        df = pd.DataFrame(
            data=np.array(columns).T,
            columns=list(used_keys)
        )

        df = df.round(4)
        print(df.to_latex(index=False))
        
    print_table(True)
    print_table(False)


def best_agreement(records):
    records = sorted(records, key=lambda x: x.split_id)
    rows = []
    for r in records:
        rows.append(r.find_best_agreement_epoch())
        
    df = pd.DataFrame(
        data=np.array(rows),
        columns=["Epoch", "Agreement Indicator", "Test Accuracy"],
    )
    df = df.round(4)
    print(df.to_latex(index=False))

