import unittest
import tensorflow as tf
import numpy as np
import yaml
import zipfile
import os
import shutil


from src.test_retest.regularizer import \
    js_divergence, batch_divergence, l1_mean

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
        with open("data/{}/train_groups.csv".format(record_label), 'r') as f:
            for line in f:
                paths = line.strip().split("\t")
                file_names = [os.path.split(p)[-1].split(".")[0] for p in paths]

                groups.append(file_names)

        # Unzip folder
        folder_path = "produced_data/{}/train_{}.zip".format(
            record_label, epoch
        )
        dest_path = "produced_data/{}/train_{}".format(
            record_label, epoch
        )

        shutil.unpack_archive(folder_path, dest_path, 'zip')

        dic = {
            "l1": [],
            "l2": [],
            "js": []
        }
        norms = []
        for name_1, name_2 in groups:
            enc_1 = np.load(dest_path + "/{}.npy".format(name_1))[-diag_dim:]
            enc_2 = np.load(dest_path + "/{}.npy".format(name_2))[-diag_dim:]
            d = numpy_utils.js_divergence(
                numpy_utils.softmax(enc_1),
                numpy_utils.softmax(enc_2)
            )
            dic["js"].append(d)
            dic["l2"].append(numpy_utils.l2_sq_mean_reg(enc_1 - enc_2))
            dic["l1"].append(numpy_utils.l1_mean_reg(enc_1 - enc_2))
            norms.append(np.linalg.norm(enc_1 - enc_2))

        for k, v in dic.items():
            print("{}: {}".format(k, np.mean(v)))
        print(np.mean(norms))

        shutil.rmtree(dest_path)

    def test_train_hidden_reg_loss(self):
        with open("tests/configs/test_train_hidden_reg_loss.yaml", "r") as f:
            config = yaml.load(f)

        for label in config["record_labels"]:
            epochs = config["record_labels"][label]["epochs"]
            diag_dim = config["record_labels"][label]["diag_dim"]
            for e in epochs:
                self.compute_js_divergence(label, e, diag_dim)

    def test_weighted_loss(self):
        x = tf.placeholder(shape=[4, 2], dtype=tf.float32)
        y = tf.placeholder(shape=[4, 2], dtype=tf.float32)

        loss = l1_mean(x - y)

        np_x = np.array([
            [1, 1],
            [2, 2],
            [3, 2],
            [9, 0]
        ])

        np_y = np.array([
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0]
        ])

        np_loss = np.mean(np.abs(np_x - np_y))

        tf_loss = self.sess.run(loss, {
            x: np_x,
            y: np_y
        })

        self.assertTrue(np.allclose(np_loss, tf_loss))

        np_weights = np.array([
            [1],
            [0],
            [0],
            [1]
        ])

        weights = tf.placeholder(shape=[4, 1], dtype=tf.float32)

        loss = l1_mean(x - y, weights)
        loss_5 = l1_mean(x - y, 5 * weights)
        loss_0 = l1_mean(x - y, 0 * weights)
        tf_loss, tf_loss_5, tf_loss_0 = self.sess.run([loss, loss_5, loss_0], {
            x: np_x,
            y: np_y,
            weights: np_weights
        })

        self.assertTrue(np.isclose(tf_loss * 5, tf_loss_5))
        self.assertTrue(np.isclose(tf_loss * 0, tf_loss_0))


if __name__ == "__main__":
    unittest.main()
