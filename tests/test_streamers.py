import unittest
import yaml
import importlib
from src.data.streaming.mri_streaming import \
    MRISingleStream, MRIDiagnosePairStream, MRISamePatientSameAgePairStream

import subprocess


def str_to_module(s):
    mod_name = ".".join(s.split(".")[:-1])
    mod = importlib.import_module(mod_name)
    f = getattr(mod, s.split(".")[-1])
    return f


def qualified_path(_class):
    return _class.__module__ + "." + _class.__name__


def create_object(config):
    _class = config["class"]
    _class = str_to_module(_class)
    _params = config["params"]
    return _class(**_params)


class TestReproducibility(unittest.TestCase):
    def setUp(self):
        with open("tests/configs/test_streamer.yaml") as f:
            config = yaml.load(f)

        self.config = config

    def groups_are_equal(self, groups_1, groups_2):
        self.assertEqual(len(groups_1), len(groups_2))
        for g1, g2 in zip(groups_1, groups_2):
            fids1 = g1.file_ids
            fids2 = g2.file_ids
            self.assertEqual(len(fids1), len(fids2))
            for f1, f2 in zip(fids1, fids2):
                self.assertEqual(f1, f2)

    def reproducibility_accross_runs(self):
        config_path = "tests/tmp_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

        cmd = ["python", "-m", "tests.run_streamer", config_path]
        oi = subprocess.check_output(cmd)

        for i in range(5):
            o = subprocess.check_output(cmd)
            self.assertEqual(oi, o)

    def reproducibility_within_run(self):
        streamer = create_object(self.config)
        self.assertTrue(streamer is not None)

        i_train_groups = [g for g in streamer.groups if g.is_train]
        i_test_groups = [g for g in streamer.groups if not g.is_train]

        # Should always produce the same split and groups
        for i in range(5):
            streamer = create_object(self.config)
            self.assertTrue(streamer is not None)

            train_groups = [g for g in streamer.groups if g.is_train]
            test_groups = [g for g in streamer.groups if not g.is_train]

            self.groups_are_equal(i_train_groups, train_groups)
            self.groups_are_equal(i_test_groups, test_groups)

    def test_mri_diagnose_pair(self):
        self.config["class"] = qualified_path(MRIDiagnosePairStream)
        self.reproducibility_within_run()
        self.reproducibility_accross_runs()

    def test_mri_single_stream(self):
        self.config["class"] = qualified_path(MRISingleStream)
        self.reproducibility_within_run()
        self.reproducibility_accross_runs()

    def test_mri_same_patient_same_age_pair_stream(self):
        self.config["class"] = qualified_path(MRISamePatientSameAgePairStream)
        self.reproducibility_within_run()
        self.reproducibility_accross_runs()


if __name__ == "__main__":
    unittest.main()
