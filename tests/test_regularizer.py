import unittest
import tensorflow as tf
import numpy as np
import yaml
import zipfile
import os
import shutil


from src.test_retest.regularizer import \
    js_divergence, batch_divergence

from src.test_retest import numpy_utils


class TestRegularizers(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()

    def tearDown(self):
        self.sess.close()

    def test_js_divergence(self):
        d = 20
        n = 10000
        A = np.random.rand(n, 2 * d)

        x1 = tf.placeholder(
            dtype=tf.float64,
            shape=[n, d],
            name="x1"
        )

        x2 = tf.placeholder(
            dtype=tf.float64,
            shape=[n, d],
            name="x2"
        )

        y = batch_divergence(
            x1,
            x2,
            d,
            js_divergence
        )

        res = self.sess.run(
            y,
            {
                x1: A[:, :d],
                x2: A[:, d:]
            }
        )

        comp = numpy_utils.batch_divergence(
            A[:, :d],
            A[:, d:],
            numpy_utils.js_divergence
        )

        self.assertTrue(np.allclose(res, comp))

    def compute_js_divergence(self, record_label, epoch, diag_dim):
        print("Record {}, epoch {}".format(record_label, epoch))
        # Load groups
        groups = []
        with open("data/{}/test_groups.csv".format(record_label), 'r') as f:
            for line in f:
                paths = line.strip().split("\t")
                file_names = [os.path.split(p)[-1].split(".")[0] for p in paths]

                groups.append(file_names)

        # Unzip folder
        folder_path = "produced_data/{}/test_{}.zip".format(
            record_label, epoch
        )
        dest_path = "produced_data/{}/test_{}".format(
            record_label, epoch
        )

        shutil.unpack_archive(folder_path, dest_path, 'zip')

        ds = []
        norms = []
        for name_1, name_2 in groups:
            enc_1 = np.load(dest_path + "/{}.npy".format(name_1))[-diag_dim:]
            enc_2 = np.load(dest_path + "/{}.npy".format(name_2))[-diag_dim:]
            d = numpy_utils.js_divergence(
                numpy_utils.softmax(enc_1),
                numpy_utils.softmax(enc_2)
            )
            ds.append(d)
            norms.append(np.linalg.norm(enc_1 - enc_2))

        print(np.mean(ds))
        print(np.mean(norms))

        shutil.rmtree(dest_path)

    def test_test_hidden_reg_loss(self):
        with open("tests/configs/test_test_hidden_reg_loss.yaml", "r") as f:
            config = yaml.load(f)

        for label in config["record_labels"]:
            epochs = config["record_labels"][label]["epochs"]
            diag_dim = config["record_labels"][label]["diag_dim"]
            for e in epochs:
                self.compute_js_divergence(label, e, diag_dim)


if __name__ == "__main__":
    unittest.main()
