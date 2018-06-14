import unittest
import os
import numpy as np
from src.test_retest.numpy_utils import js_divergence


class TestJSDivergence(unittest.TestCase):
    def test_ae_embeddings(self):
        smt_label = "20180613-123945-debug"
        group_path = os.path.join("data", smt_label, "test_groups.csv")
        feature_path = os.path.join("produced_data", smt_label, "test_0")

        def load_vec(fid):
            file_name = os.path.split(fid)[-1].split(".")[0]
            return np.load(os.path.join(feature_path, file_name + ".npy"))

        test = []
        retest = []
        with open(group_path, "r") as f:
            for line in f:
                fid1, fid2 = line.strip().split("\t")
                test.append(load_vec(fid1))
                retest.append(load_vec(fid2))

        def softmax(vec):
            return np.exp(vec) / np.sum(np.exp(vec))

        divs = []
        for vec1, vec2 in zip(test, retest):
            js = js_divergence(softmax(vec1), softmax(vec2))
            divs.append(js)

        print("js loss {}".format(np.mean(divs)))


if __name__ == "__main__":
    unittest.main()
