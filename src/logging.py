import json


class MetricLogger(object):
    def __init__(self, outdir, description="description goes here"):
        self.description = description
        self.metrics = {}
        self.outdir = outdir

    def add_metric(self, label, value):
        if label not in self.metrics:
            self.metrics[label] = []

        self.metrics[label].append(float(value))

    def add_evaluations(self, evaluation_dic, exclude=""):
        for k in evaluation_dic:
            if k == exclude:
                continue

            self.add_metric(k, evaluation_dic[k])

    def dump(self):
        metrics = {}
        for k in self.metrics:
            det = {
                "x_label": "global step",
                "x": int(self.metrics["global_step"]),
                "y": self.metrics[k]
            }
            metrics[k] = det

        obj = {
            "text_outcome": self.description,
            "numeric_outcome": metrics
        }

        with open("%s/sumatra_outcome.json" % (self.outdir), "w") as outfile:
            json.dump(obj, outfile)

    def to_json(self):
        obj = {
            "text_outcome": self.description,
            "numeric_outcome": self.metrics
        }

        return json.dumps(obj.__dict__)
