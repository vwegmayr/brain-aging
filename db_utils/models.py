import os
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools


def extract_tags(tag_string):
    """
    Splits input string on commata.
    """
    tags = tag_string.strip().split(",")
    return set(tags)


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

    def get_best_validation_epoch(self, metric="acc"):
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
    def __init__(self, records, group_label):
        self.group_label = group_label
        # Assign run IDs
        # Sort records based on label
        self.records = sorted(
            records,
            key=lambda r: r.label,
            reverse=False
        )

        for i in range(len(records)):
            self.records[i].run_id = i + 1

    @staticmethod
    def get_markers():
        return itertools.cycle(('', '+', '.', 'o', '*')) 

    @staticmethod
    def get_colors():
        return iter(plt.cm.tab10(np.linspace(0, 1, 11)))

    def plot_group(self, metric, x_label, y_label, legend_loc, type="line"):
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
                marker=next(markers)
            )

        plt.xlabel(x_label)
        plt.ylabel(y_label)

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
            # mean line
            plt.errorbar(
                x,
                stats.get_means(),
                yerr=stats.get_stds(),
                c=c,
                marker=m,
                label=group.group_label,
                linestyle='-.',
            )
            # median line
            plt.plot(
                x,
                stats.get_medians(),
                c=c,
                marker=m,
                label=group.group_label,
                linestyle=':',
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
