import os
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from collections import OrderedDict


def extract_tags(tag_string):
    """
    Splits input string on commata.
    """
    tags = tag_string.strip().split(",")
    return set(tags)


def toCamelCase(word):
    return ''.join(x.capitalize() or '_' for x in word.split('_'))


class Record(object):
    """
    Representation of a Sumatra record.
    """
    def __init__(self, dic):
        """
        Arg:
            - dic: dictionary representing a row from
              the sumatra 'django_store_record' table
        """
        self.label = dic["label"]
        self.reason = dic["reason"]
        self.timestamp = dic["timestamp"]
        self.tags = extract_tags(dic["tags"])
        self.params_id = int(dic["parameters_id"])
        self.version = dic["version"]
        self.config = None
        self.run_id = -1  # identify records within CV group

    def load_metrics(self, data_path):
        """
        Loads the 'sumatra_outcome.json' file corresponding
        to this record.

        Arg:
            - data_path: path to folder containg records
        """
        path = os.path.join(data_path, self.label, "sumatra_outcome.json")
        with open(path, 'r') as f:
            data = json.load(f)

        self.metrics = data["numeric_outcome"]
        return self.metrics

    def get_metric_values(self, metric, epoch=-1):
        values = self.metrics[metric]["y"]
        if epoch >= 0:
            return values[epoch]
        else:
            return values

    def get_best_validation_epoch(self, metric="validation_acc"):
        values = self.get_metric_values(metric, epoch=-1)
        return np.argmax(values)

    def find_tag(self, partial_tag):
        """
        Checks if this records as a tag containing the
        input string 'partial_tag'.

        Arg:
            - partial_tag: string being part of a tag

        Return:
            - tag containg 'partial_tag' or "NA" if 'partial_tag'
              could not be matched

        """
        for t in self.tags:
            if t.startswith(partial_tag):
                return t

        return "NA"

    def __str__(self):
        return self.label


class RecordGroup(object):
    def __init__(self, records, group_label, data_path):
        self.group_label = group_label
        # Assign run IDs
        # Sort records based on label
        self.records = sorted(
            records,
            key=lambda r: r.label,
            reverse=False
        )

        for r in self.records:
            r.load_metrics(data_path)

        for i in range(len(records)):
            self.records[i].run_id = i + 1

    @staticmethod
    def get_markers():
        return itertools.cycle(('x', '.', 'o', '*')) 

    @staticmethod
    def get_colors():
        return iter(plt.cm.tab10(np.linspace(0, 1, 11)))

    def plot_group(self, metric, x_label, y_label, legend_loc, type="line"):
        self.print_epoch_stats(metric)

        plt.figure()
        colors = RecordGroup.get_colors()
        markers = RecordGroup.get_markers()
        for record in self.records:
            values = record.get_metric_values(metric)
            x = np.array(list(range(len(values)))) + 1
            label = "run " + str(record.run_id)
            plt.plot(
                x,
                values,
                linewidth=2,
                label=label,
                c=next(colors),
                marker=next(markers),
                markersize=10,
            )

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(self.group_label)
        plt.legend(loc=legend_loc, ncol=1)
        plt.grid(True)
        plt.show()

    def group_stats_per_epoch(self, metric):
        all_values = []
        for record in self.records:
            values = record.get_metric_values(metric)
            all_values.append(values)

        all_values = np.array(all_values)
        return EpochStats(all_values)

    def print_epoch_stats(self, metric):
        # Per epoch mean
        stats = self.group_stats_per_epoch(metric)
        # create table with fromat
        # epoch | mean | std | median | percentile
        header = ["epoch", "mean", "std", "median", "5-percentile",
                  "95-percentile"]
        table = []
        n_epochs = stats.n_epochs
        for i in range(n_epochs):
            row = [
                i,
                stats.get_means()[i],
                stats.get_stds()[i],
                stats.get_medians()[i],
                stats.get_percentile(
                    percentile=5,
                    epoch=i
                ),
                stats.get_percentile(
                    percentile=95,
                    epoch=i
                ),
            ]

            table.append(row)

        df = pd.DataFrame(
            data=np.array(table),
            columns=header
        )
        print(self.group_label)
        self.print_df(df)

    def print_df(self, df):
        df = df.round(4)
        print(df)
        print(df.to_latex(index=False))

    def print_run_accuracies(self, metrics):
        # Fold accuracy based on best validation epoch
        header = ["run", "bestValEpoch", "bestValAcc"]

        metric_to_values = OrderedDict()
        for m in metrics:
            header.append(toCamelCase(m))
            metric_to_values[m] = []

        table = []
        for record in self.records:
            run = record.run_id
            best_ep = record.get_best_validation_epoch(
                "validation_acc"
            )
            best_val_acc = record.get_metric_values(
                metric="validation_acc",
                epoch=best_ep
            )
            """
            test_acc = record.get_metric_values(
                metric="test_acc",
                epoch=best_ep
            )
            """
            row = [run, best_ep, best_val_acc]

            for m in metrics:
                val = record.get_metric_values(
                    metric=m,
                    epoch=best_ep
                )
                row.append(val)
                metric_to_values[m].append(val)

            table.append(row)

        table = np.array(table)

        print(self.group_label)
        df = pd.DataFrame(
            data=table,
            columns=header
        )
        self.print_df(df)

        # Overall accuracy
        self.print_overall_metric(metric_to_values)

    def print_overall_metric(self, metric_to_values):
        header = ["metricName", "mean", "std", "median", "5-percentile",
                  "95-percentile"]

        table = []
        prec = 4
        for k, values in metric_to_values.items():
            row = [
                toCamelCase(k),
                round(np.mean(values), prec),
                round(np.std(values), prec),
                round(np.median(values), prec),
                round(np.percentile(values, 5), prec),
                round(np.percentile(values, 95), prec)
            ]
            table.append(row)

        df = pd.DataFrame(
            data=np.array(table),
            columns=header
        )
        # print(self.group_label + " " + metric_name)
        self.print_df(df)

    @staticmethod
    def compare_groups(groups, metric, x_label, y_label):
        colors = RecordGroup.get_colors()
        markers = RecordGroup.get_markers()

        plt.figure()
        for group in groups:
            stats = group.group_stats_per_epoch(metric)
            x = np.array(list(range(stats.n_epochs))) + 1
            c = next(colors)
            m = next(markers)
            # mean line with error bars
            plt.errorbar(
                x,
                stats.get_means(),
                yerr=stats.get_stds(),
                c=c,
                marker=m,
                label=group.group_label,
                linestyle='--',
                elinewidth=1,
                capsize=10
            )
            # median line
            plt.plot(
                x,
                stats.get_medians(),
                c=c,
                marker=m,
                #label=group.group_label,
                linewidth=0,
                markersize=13
            )

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.legend(loc=0, ncol=1)
        plt.grid(True)
        plt.show()


class EpochStats(object):
    def __init__(self, values):
        self.values = values
        self.means = np.mean(values, axis=0)
        self.stds = np.std(values, axis=0)
        self.medians = np.median(values, axis=0)
        self.n_epochs = values.shape[1]

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds

    def get_medians(self):
        return self.medians

    def get_percentile(self, percentile, epoch):
        epoch_values = self.values[:, epoch]
        return np.percentile(epoch_values, percentile)
