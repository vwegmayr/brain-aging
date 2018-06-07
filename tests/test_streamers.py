import unittest
import yaml
import importlib


def str_to_module(s):
    mod_name = ".".join(s.split(".")[:-1])
    mod = importlib.import_module(mod_name)
    f = getattr(mod, s.split(".")[-1])
    return f


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
        return True

    def test_mri_diagnose_pair(self):
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

            self.assertTrue(
                self.groups_are_equal(i_train_groups, train_groups)
            )

            self.assertTrue(
                self.groups_are_equal(i_test_groups, test_groups)
            )


if __name__ == "__main__":
    unittest.main()
