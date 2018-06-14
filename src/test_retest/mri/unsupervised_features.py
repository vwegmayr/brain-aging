from radiomics import featureextractor
import SimpleITK as sitk
import os
import numpy as np
import json
from memory_profiler import profile
import gc
import tensorflow as tf
from subprocess import call, Popen
import math
from sklearn.decomposition import IncrementalPCA
from functools import reduce

from modules.models.data_transform import DataTransformer
from src.test_retest.test_retest_base import EvaluateEpochsBaseTF
from src.test_retest.test_retest_base import linear_trafo
from src.test_retest.test_retest_base import regularizer
from src.test_retest.test_retest_base import mnist_input_fn
from src.test_retest.non_linear_estimator import name_to_hidden_regularization
from src.train_hooks import BatchDumpHook, RobustnessComputationHook, \
    SumatraLoggingHook, LogisticPredictionHook


class PyRadiomicsFeatures(DataTransformer):
    def __init__(self, streamer):
        # Initialize streamer
        _class = streamer["class"]
        self.streamer = _class(**streamer["params"])

    def get_extractor(self):
        # Initialize extractor
        extractor = featureextractor.RadiomicsFeaturesExtractor()
        extractor.enableAllImageTypes()
        extractor.enableAllFeatures()

        return extractor

    def transform(self, X, y=None):
        out_path = os.path.join(self.save_path, "features")
        os.mkdir(out_path)
        # Stream image one by one
        batches = self.streamer.get_batches()
        for batch in batches:
            for group in batch:
                for file_id in group.get_file_ids():
                    path = self.streamer.get_file_path(file_id)

                    sitk_im = sitk.ReadImage(path)
                    all_ones = np.ones(sitk_im.GetSize())
                    sitk_mask = sitk.GetImageFromArray(all_ones)

                    extractor = self.get_extractor()
                    features = extractor.computeFeatures(sitk_im, sitk_mask, "brain")

                    with open(
                        os.path.join(out_path, str(file_id) + ".json"),
                        "w"
                    ) as f:
                        json.dump(features, f, indent=2)


class PyRadiomicsFeaturesSpawn(DataTransformer):
    def __init__(self, streamer, out_dir, n_processes):
        # Initialize streamer
        _class = streamer["class"]
        self.streamer = _class(**streamer["params"])
        self.out_dir = out_dir
        self.n_processes = n_processes

    def get_extractor(self):
        # Initialize extractor
        extractor = featureextractor.RadiomicsFeaturesExtractor()
        extractor.enableAllImageTypes()
        extractor.enableAllFeatures()

        return extractor

    def transform(self, X, y=None):
        out_path = self.out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Stream image one by one
        batches = self.streamer.get_batches()
        processes = []
        for batch in batches:
            for group in batch:
                for file_id in group.get_file_ids():
                    image_label = self.streamer.get_image_label(file_id)
                    path_in = self.streamer.get_file_path(file_id)
                    path_out = os.path.join(out_path, str(image_label) + ".json")

                    cmd = 'python -m src.test_retest.mri.run_py_radiomics_transformer '
                    cmd += "{} {}".format(path_in, path_out)
                    print(cmd)
                    # call(cmd, shell=True)
                    proc = Popen(cmd, shell=True)
                    processes.append(proc)

                    if len(processes) >= self.n_processes:
                        for p in processes:
                            p.wait()
                        processes = []

        self.streamer = None


class PyRadiomicsSingleFileTransformer(DataTransformer):
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path = out_path

    def get_extractor(self):
        # Initialize extractor
        extractor = featureextractor.RadiomicsFeaturesExtractor()
        extractor.enableAllImageTypes()
        extractor.enableAllFeatures()

        return extractor

    def transform(self, X, y=None):
        sitk_im = sitk.ReadImage(self.in_path)
        all_ones = np.ones(sitk_im.GetSize())
        sitk_mask = sitk.GetImageFromArray(all_ones)

        extractor = self.get_extractor()
        features = extractor.computeFeatures(sitk_im, sitk_mask, "brain")

        with open(
            os.path.join(self.out_path),
            "w"
        ) as f:
            json.dump(features, f, indent=2)


class MriIncrementalPCA(DataTransformer):
    """
    batch_size has to be larger than n_components
    """
    def __init__(self, streamer, n_components):
        _class = streamer["class"]
        self.streamer = _class(**streamer["params"])
        self.pca = IncrementalPCA(
            n_components=n_components,
            batch_size=self.streamer.batch_size
        )

    def transform(self, X, y=None):
        batches = self.streamer.get_batches()

        for batch in batches:
            file_ids = [fid for group in batch for fid in group.file_ids]
            X = []
            for fid in file_ids:
                path = self.streamer.get_file_path(fid)
                im = self.streamer.load_sample(path)
                X.append(im.ravel())

            X = np.array(X)
            print(X.shape)
            self.pca = self.pca.partial_fit(X)

        self.streamer = None


class PCAAutoEncoder(EvaluateEpochsBaseTF):
    def model_fn(self, features, labels, mode, params):
        input_dim = params["input_dim"]
        X = tf.reshape(
            features["X_0"],
            [-1, input_dim]
        )

        hidden_dim = params["hidden_dim"]
        w = tf.get_variable(
            name="weights",
            shape=[input_dim, hidden_dim],
            dtype=X.dtype,
            initializer=tf.contrib.layers.xavier_initializer(seed=43)
        )

        hidden = tf.matmul(X, w, name="hidden_rep")
        # hidden = tf.nn.sigmoid(hidden)
        w_T = tf.transpose(w)
        reconstruction = tf.matmul(
            hidden,
            w_T,
            name="reconstruction"
        )

        predictions = {
            "hidden_rep": hidden,
            "reconstruction": reconstruction
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        # Compute loss
        loss = tf.losses.mean_squared_error(X, reconstruction)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(loss, tf.train.get_global_step())

        dump_hook_train, dump_hook_test = \
            self.get_batch_dump_hook(hidden, features["file_name_0"])

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[dump_hook_train]
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            evaluation_hooks=[dump_hook_test]
        )

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return self.streamer.get_input_fn(train)


class PCAAutoEncoderTuples(EvaluateEpochsBaseTF):
    def normalize_single_image(self, im):
        mean, var = tf.nn.moments(im, axes=[0])
        std = tf.sqrt(var)
        std += 0.001
        return (im - mean) / std

    def normalize_image_batch(self, batch, voxel_means, voxel_stds):
        # normalize every image
        batch = tf.map_fn(
            self.normalize_single_image,
            batch
        )

        # voxel normalization computed across train set
        batch = (batch - voxel_means) / voxel_stds

        # normalize every image
        batch = tf.map_fn(
            self.normalize_single_image,
            batch
        )

        return batch

    def model_fn(self, features, labels, mode, params):
        input_dim = params["input_dim"]
        x_0 = tf.reshape(
            features["X_0"],
            [-1, input_dim]
        )
        x_1 = tf.reshape(
            features["X_1"],
            [-1, input_dim]
        )

        if params["normalize_images"]:
            voxel_means = tf.constant(
                self.streamer.get_voxel_means(),
                dtype=x_0.dtype
            )
            voxel_means = tf.reshape(voxel_means, [-1, input_dim])

            voxel_stds = tf.constant(
                self.streamer.get_voxel_stds(),
                dtype=x_0.dtype
            )
            voxel_stds = tf.reshape(voxel_stds, [-1, input_dim])

            x_0 = self.normalize_image_batch(x_0, voxel_means, voxel_stds)
            x_1 = self.normalize_image_batch(x_1, voxel_means, voxel_stds)

        hidden_dim = params["hidden_dim"]
        w = tf.get_variable(
            name="weights",
            shape=[input_dim, hidden_dim],
            dtype=x_0.dtype,
            initializer=tf.contrib.layers.xavier_initializer(seed=43)
        )

        hidden_0 = tf.matmul(x_0, w, name="hidden_rep_0")
        hidden_1 = tf.matmul(x_1, w, name="hidden_rep_1")

        rec_0 = tf.matmul(
            hidden_0,
            tf.transpose(w),
            name="reconstruction_0"
        )

        rec_1 = tf.matmul(
            hidden_1,
            tf.transpose(w),
            name="reconstruction_1"
        )

        predictions = {
            "hidden_rep": hidden_0,
            "reconstruction": rec_0
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        # Compute loss
        # loss = tf.reduce_sum(tf.square(input_mri - reconstruction))
        if params["asymmetric"]:
            loss_0 = tf.losses.mean_squared_error(x_0, rec_1)
            loss_1 = tf.losses.mean_squared_error(x_1, rec_0)
        else:
            loss_0 = tf.losses.mean_squared_error(x_0, rec_0)
            loss_1 = tf.losses.mean_squared_error(x_1, rec_1)
        reconstruction_loss = loss_0 / 2 + loss_1 / 2

        # Regularization
        to_reg_0 = hidden_0
        to_reg_1 = hidden_1
        diagnose_dim = params["diagnose_dim"]
        if diagnose_dim > 0:
            patient_dim = hidden_dim - diagnose_dim
            patient_encs_0, diag_encs_0 = tf.split(
                hidden_0,
                [patient_dim, diagnose_dim],
                axis=1
            )

            patient_encs_1, diag_encs_1 = tf.split(
                hidden_1,
                [patient_dim, diagnose_dim],
                axis=1
            )

            to_reg_0 = diag_encs_0
            to_reg_1 = diag_encs_1

        reg_loss = tf.constant(0, dtype=tf.int32, shape=None)
        reg_lambda = params["hidden_lambda"]
        if reg_lambda != 0:
            reg_name = params["hidden_regularizer"]
            reg_loss = name_to_hidden_regularization(
                0,
                reg_name,
                to_reg_0,
                to_reg_1
            )
            reg_loss *= reg_lambda

        reg_loss = tf.cast(reg_loss, reconstruction_loss.dtype)
        loss = reconstruction_loss + reg_loss

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(loss, tf.train.get_global_step())


        # Set up hooks
        train_hooks = []
        eval_hooks = []

        train_hook_names = params["train_hooks"]
        eval_hook_names = params["eval_hooks"]

        if "embeddings" in train_hook_names:
            hidden_0_hook_train, hidden_0_hook_test = \
                self.get_batch_dump_hook(hidden_0, features["file_name_0"])
            train_hooks.append(hidden_0_hook_train)
            eval_hooks.append(hidden_0_hook_test)

            hidden_1_hook_train, hidden_1_hook_test = \
                self.get_batch_dump_hook(hidden_1, features["file_name_1"])
            train_hooks.append(hidden_1_hook_train)
            eval_hooks.append(hidden_1_hook_test)

            train_feature_folder = hidden_0_hook_train.get_feature_folder_path()
            test_feature_folder = hidden_0_hook_test.get_feature_folder_path()

        if "robustness" in train_hook_names:
            robustness_hook_train = self.get_robusntess_analysis_hook(
                feature_folder=train_feature_folder,
                train=True
            )
            robustness_hook_test = self.get_robusntess_analysis_hook(
                feature_folder=test_feature_folder,
                train=False
            )
            train_hooks.append(robustness_hook_train)
            eval_hooks.append(robustness_hook_test)

        if "predictions" in eval_hook_names:
            prediction_hook = LogisticPredictionHook(
                train_folder=train_feature_folder,
                test_folder=test_feature_folder,
                streamer=self.streamer,
                model_save_path=self.save_path,
                out_dir=self.data_params["dump_out_dir"],
                epoch=self.current_epoch,
                target_label="healthy"
            )
            eval_hooks.append(prediction_hook)

        # log embedding loss
        log_hook_train = SumatraLoggingHook(
            ops=[reg_loss, reconstruction_loss],
            names=["hidden_reg_loss", "reconstruction_loss"],
            logger=self.metric_logger,
            namespace="train"
        )
        train_hooks.append(log_hook_train)

        log_hook_test = SumatraLoggingHook(
            ops=[reg_loss, reconstruction_loss],
            names=["hidden_reg_loss", "reconstruction_loss"],
            logger=self.metric_logger,
            namespace="test"
        )
        eval_hooks.append(log_hook_test)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            evaluation_hooks=eval_hooks
        )

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return self.streamer.get_input_fn(train)


class MnistPCAAutoEncoder(PCAAutoEncoder):
    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return mnist_input_fn(
            X,
            self.data_params,
            input_fn_config=input_fn_config
        )


class RandomImageAutoEncoder(PCAAutoEncoder):
    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        # images = np.random.rand(2000, 28, 28)
        images = np.zeros((2000, 28, 28))
        labels = np.ones((2000, 1))

        return tf.estimator.inputs.numpy_input_fn(
            x={"X_0": images},
            y=labels,
            **input_fn_config,
        )


# Source: https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py
class Conv2DAutoEncoder(EvaluateEpochsBaseTF):
    def model_fn(self, features, labels, mode, params):
        input_dim = params["input_dim"]
        x = features["X_0"]

        n_filters = params["n_filters"]
        filter_sizes = params["filter_sizes"]

        x = tf.reshape(x, [-1, input_dim, input_dim, 1])
        current_input = x

        # Build the encoder
        encoder = []
        shapes = []
        for layer_i, n_output in enumerate(n_filters):
            n_input = current_input.get_shape().as_list()[3]
            shapes.append(current_input.get_shape().as_list())
            W_shape = [
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input,
                n_output
            ]
            W = tf.get_variable(
                name="filters_layer_" + str(layer_i),
                shape=W_shape,
                initializer=tf.contrib.layers.xavier_initializer(seed=40)
            )

            b = tf.get_variable(
                name="bias_layer_" + str(layer_i),
                shape=[n_output],
                initializer=tf.initializers.zeros
            )
            encoder.append(W)
            output = tf.nn.relu(
                tf.add(
                    tf.nn.conv2d(
                        current_input, W, strides=[1, 2, 2, 1], padding='SAME'
                    ),
                    b
                )
            )
            current_input = output

            #current_input = tf.layers.max_pooling2d(current_input, 2, 1)

        z = current_input
        encoder.reverse()
        shapes.reverse()
        # Build the decoder
        for layer_i, shape in enumerate(shapes):
            # deconv
            if not params["tied_weights"]:
                W_shape = encoder[layer_i].get_shape().as_list()
                print(shape)
                print(W_shape)
                W = tf.get_variable(
                    name="filters_deconv_layer_" + str(layer_i),
                    shape=[W_shape[0], W_shape[1], W_shape[2], W_shape[3]],
                    initializer=tf.contrib.layers.xavier_initializer(seed=40)
                )
            else:
                W = encoder[layer_i]
            b = tf.get_variable(
                name="bias_deconv_layer_" + str(layer_i),
                shape=W.get_shape().as_list()[2],
                initializer=tf.initializers.zeros
            )

            input_shapes = current_input.get_shape().as_list()
            print("input shape {}".format(input_shapes))
            print("W {}".format(W))
            output = tf.nn.relu(
                tf.add(
                    tf.nn.conv2d_transpose(
                        current_input, W,
                        tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                        strides=[1, 2, 2, 1], padding='SAME'
                    ),
                    b
                )
            )
            print("out shape {}".format(output.get_shape().as_list()))
            current_input = output

        y = current_input
        print("y {}".format(y))
        predictions = {
            "hidden_rep": z,
            "reconstruction": y
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        loss = tf.losses.mean_squared_error(x, y)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(loss, tf.train.get_global_step())

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss
        )


class Conv3DAutoEncoder(EvaluateEpochsBaseTF):
    def model_fn(self, features, labels, mode, params):
        input_shape = params["input_shape"]
        x = features["X_0"]

        n_filters = params["n_filters"]
        filter_sizes = params["filter_sizes"]

        x = tf.reshape(
            x,
            [-1, input_shape[0], input_shape[1], input_shape[2], 1]
        )
        current_input = x

        # Build the encoder
        encoder = []
        shapes = []
        for layer_i, n_output in enumerate(n_filters):
            # print("layer {}".format(layer_i))
            n_input = current_input.get_shape().as_list()[4]
            shapes.append(current_input.get_shape().as_list())
            W_shape = [
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input,
                n_output
            ]
            W = tf.get_variable(
                name="filters_layer_" + str(layer_i),
                shape=W_shape,
                initializer=tf.contrib.layers.xavier_initializer(seed=40)
            )
            b = tf.get_variable(
                name="bias_layer_" + str(layer_i),
                shape=[n_output],
                initializer=tf.initializers.zeros
            )

            encoder.append(W)
            output = tf.nn.relu(
                tf.add(
                    tf.nn.conv3d(
                        current_input, W, strides=[1, 2, 2, 2, 1], padding='SAME'
                    ),
                    b
                )
            )
            current_input = output

        z = current_input
        print("curr {}".format(current_input))
        # Non-convolutional layer, if encoding size is still to large
        dim_list = current_input.get_shape().as_list()[1:]
        cur_dim = reduce(lambda x, y: x * y, dim_list)

        if cur_dim > params["encoding_dim"]:
            # print("Non conv layer needed")
            current_input = tf.contrib.layers.flatten(current_input)
            W = tf.get_variable(
                "non_conv_w",
                shape=[cur_dim, params["encoding_dim"]],
                initializer=tf.contrib.layers.xavier_initializer(seed=40)
            )
            b = tf.get_variable(
                "non_conv_b",
                shape=[1, params["encoding_dim"]],
                initializer=tf.initializers.zeros
            )

            current_input = tf.add(
                tf.nn.relu(tf.matmul(current_input, W)),
                b
            )

            z = current_input

            if not params["tied_weights"]:
                W = tf.get_variable(
                    "non_conv_w_dec",
                    shape=[cur_dim, params["encoding_dim"]],
                    initializer=tf.contrib.layers.xavier_initializer(seed=40)
                )

            b = tf.get_variable(
                "non_conv_b_dec",
                shape=[1, cur_dim],
                initializer=tf.initializers.zeros
            )

            current_input = tf.add(
                tf.nn.relu(tf.matmul(current_input, tf.transpose(W))),
                b
            )
            # Unflatten
            current_input = tf.reshape(current_input, [-1] + dim_list)

        encoder.reverse()
        shapes.reverse()
        # Build the decoder
        for layer_i, shape in enumerate(shapes):
            # deconv
            W = encoder[layer_i]
            b = tf.get_variable(
                name="bias_deconv_layer_" + str(layer_i),
                shape=W.get_shape().as_list()[3],
                initializer=tf.initializers.zeros
            )

            output = tf.nn.relu(
                tf.add(
                    tf.nn.conv3d_transpose(
                        current_input, W,
                        tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3], shape[4]]),
                        strides=[1, 2, 2, 2, 1], padding='SAME'
                    ),
                    b
                )
            )
            current_input = output

        y = current_input

        predictions = {
            "hidden_rep": z,
            "reconstruction": y
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        loss = tf.losses.mean_squared_error(x, y)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(loss, tf.train.get_global_step())

        y_hook_train, y_hook_test = \
            self.get_batch_dump_hook(z, features["file_name_0"])

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[y_hook_train]
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            evaluation_hooks=[y_hook_test]
        )

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return self.streamer.get_input_fn(train)


class MnistConv2DAutoEncoder(Conv2DAutoEncoder):
    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return mnist_input_fn(
            X,
            self.data_params,
            input_fn_config=input_fn_config
        )


class MnistConv3DAutoEncoder(Conv3DAutoEncoder):
    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return mnist_input_fn(
            X,
            self.data_params,
            input_fn_config=input_fn_config
        )
