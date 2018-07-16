import os
import json


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
