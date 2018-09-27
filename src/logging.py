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

    def add_evaluations(self, namespace, evaluation_dic, exclude=""):
        for k in evaluation_dic:
            if k == exclude:
                continue

            if namespace is not None:
                self.add_metric(namespace + "_" + k, evaluation_dic[k])
            else:
                self.add_metric(k, evaluation_dic[k])

    def dump(self):
        metrics = {}
        for k in self.metrics:
            n = len(self.metrics[k])
            det = {
                "x_label": "i-th evaluation",
                "x": list(range(1, n + 1)),
                "y": self.metrics[k]
            }
            metrics[k] = det

        obj = {
            "text_outcome": self.description,
            "numeric_outcome": metrics
        }

        with open("%s/sumatra_outcome.json" % (self.outdir), "w") as outfile:
            json.dump(obj, outfile, indent=2)

    def log_hook_results(self, dic_vals):
        for name in dic_vals:
            if name not in self.metrics:
                self.metrics[name] = []

            dic_vals[name] = list(map(lambda x: float(x), dic_vals[name]))
            self.metrics[name].extend(dic_vals[name])

    def to_json(self):
        obj = {
            "text_outcome": self.description,
            "numeric_outcome": self.metrics
        }

        return json.dumps(obj.__dict__)
