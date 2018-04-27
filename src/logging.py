import json


class MetricLogger(object):
    def __init__(self, outdir, description="description goes here"):
        self.description = description
        self.metrics = {}
        self.outdir = outdir
        self.n_evals = 0

    def add_metric(self, label, value):
        if label not in self.metrics:
            self.metrics[label] = []

        self.metrics[label].append(float(value))

    def add_evaluations(self, evaluation_dic, exclude=""):
        self.n_evals += 1
        for k in evaluation_dic:
            if k == exclude:
                continue

            self.add_metric(k, evaluation_dic[k])

    def dump(self):
        metrics = {}
        for k in self.metrics:
            det = {
                "x_label": "i-th evaluation",
                "x": list(range(1, self.n_evals + 1)),
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
