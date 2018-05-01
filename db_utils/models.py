import os
import json


def extract_tags(tag_string):
    tags = tag_string.strip().split(",")
    return set(tags)


class Record(object):
    def __init__(self, dic):
        self.label = dic["label"]
        self.reason = dic["reason"]
        self.timestamp = dic["timestamp"]
        self.tags = extract_tags(dic["tags"])
        self.params_id = int(dic["parameters_id"])
        self.version = dic["version"]
        self.config = None

    def load_metrics(self, data_path):
        path = os.path.join(data_path, self.label, "sumatra_outcome.json")
        with open(path, 'r') as f:
            data = json.load(f)

        self.metrics = data["numeric_outcome"]
