import unittest
import yaml
import importlib
from src.data.streaming.mri_streaming import \
    MRISingleStream, MRIDiagnosePairStream, MRISamePatientSameAgePairStream, \
    SimilarPairStream

import subprocess
import numpy as np

N_RUNS = 10


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
        print(streamer.__class__)

        i_train_groups = [g for g in streamer.groups if g.is_train]
        i_test_groups = [g for g in streamer.groups if not g.is_train]

        # Should always produce the same split and groups
        for i in range(N_RUNS):
            print(">>>> run {}".format(i + 1))
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

    def test_similar_pair_stream(self):
        self.config["class"] = qualified_path(SimilarPairStream)
        self.reproducibility_within_run()
        # self.reproducibility_accross_runs()


class TestBalancedSplit(unittest.TestCase):
    def setUp(self):
        with open("tests/configs/test_streamer.yaml") as f:
            config = yaml.load(f)

        self.config = config

    def run_streamer(self):
        streamer = create_object(self.config)
        self.assertIsNotNone(streamer)

    def test_single_stream(self):
        self.config["class"] = qualified_path(MRISingleStream)
        self.config["params"]["stream_config"]["silent"] = False
        self.run_streamer()


class TestImageNormalization(unittest.TestCase):
    def setUp(self):
        with open("tests/configs/test_image_normalization.yaml") as f:
            config = yaml.load(f)

        self.config = config

    def test_normalization_on_few_images(self):
        streamer = create_object(self.config)

        file_ids = list(streamer.get_set_file_ids(True))
        all_images = []

        for fid in file_ids:
            p = streamer.get_file_path(fid)
            im = streamer.load_image(p)
            im = (im - np.mean(im)) / np.std(im)
            all_images.append(im)

        all_images = np.array(all_images)
        mu = np.mean(all_images, axis=0)
        s = np.std(all_images, axis=0)

        streamer_mu = streamer.voxel_means
        streamer_s = streamer.voxel_stds

        self.assertTrue(np.allclose(streamer_mu, mu, atol=0.0001))
        self.assertTrue(np.allclose(streamer_s, s, atol=0.0001))

    def test_normalization_all_images(self):
        self.config["params"]["stream_config"]["train_ratio"] = 1
        create_object(self.config)


class TestBatchOrder(unittest.TestCase):
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

    def reproducible_batches(self):
        streamer = create_object(self.config)
        self.assertTrue(streamer is not None)

        batches = streamer.get_batches(train=True)
        i_train_groups = [g for b in batches for g in b]
        batches = streamer.get_batches(train=False)
        i_test_groups = [g for b in batches for g in b]

        # Should always produce the same split and groups
        for i in range(N_RUNS):
            streamer = create_object(self.config)
            self.assertTrue(streamer is not None)

            batches = streamer.get_batches(train=True)
            train_groups = [g for b in batches for g in b]
            batches = streamer.get_batches(train=False)
            test_groups = [g for b in batches for g in b]

            self.groups_are_equal(i_train_groups, train_groups)
            self.groups_are_equal(i_test_groups, test_groups)

    def test_mri_single_stream(self):
        self.config["class"] = qualified_path(MRISingleStream)
        self.reproducible_batches()


if __name__ == "__main__":
    unittest.main()
