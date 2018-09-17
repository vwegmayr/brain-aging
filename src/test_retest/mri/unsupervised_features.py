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
import pydoc

from modules.models.data_transform import DataTransformer
from src.test_retest.test_retest_base import EvaluateEpochsBaseTF
from src.test_retest.test_retest_base import linear_trafo
from src.test_retest.test_retest_base import regularizer
from src.test_retest.test_retest_base import mnist_input_fn
from src.test_retest.mri.model_components import name_to_hidden_regularization
from src.train_hooks import BatchDumpHook, RobustnessComputationHook, \
    SumatraLoggingHook, PredictionHook, PredictionRobustnessHook
from src.train_hooks import HookFactory

from .model_components import MultiLayerPairEncoder, Conv3DEncoder
from .model_components import MultiLayerPairDecoder, Conv3DDecoder


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
                        json.dump(features, f, indent=2, ensure_ascii=False)


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
            json.dump(features, f, indent=2, ensure_ascii=False)


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
    def model_fn(self, features, labels, mode, params):
        encoder = MultiLayerPairEncoder(
            features=features,
            params=params,
            streamer=self.streamer
        )

        decoder = MultiLayerPairDecoder(
            features=features,
            params=params,
            encoder=encoder
        )

        hidden_0, hidden_1 = encoder.get_encodings()

        # Make sure embedinngs have the correct dimension
        dim = hidden_0.get_shape().as_list()[-1]
        assert dim == params["hidden_dim"]

        rec_0, rec_1 = decoder.get_nodes()

        predictions = {
            "input": encoder.get_nodes()[0],
            "encoding": hidden_0,
            "decoding": rec_0,
            "file_name": features["file_name_0"]
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        # Compute loss
        reg_loss = decoder.get_regularization_loss()
        reconstruction_loss = decoder.get_reconstruction_loss()
        loss = decoder.get_total_loss()

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(loss, tf.train.get_global_step())

        # Set up hooks
        train_hooks, eval_hooks = self.get_mri_ae_hooks(
            reg_loss=reg_loss,
            rec_loss=reconstruction_loss,
            enc_0=hidden_0,
            enc_1=hidden_1,
            features=features,
            params=params
        )

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
                        tf.stack([tf.shape(x)[0], shape[1],
                                  shape[2], shape[3]]),
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
        encoder = Conv3DEncoder(features, params, self.streamer)
        decoder = Conv3DDecoder(features, params, encoder)

        z = encoder.get_encoding()
        y = decoder.get_reconstruction()
        x = encoder.get_reconstruction_target()

        flattened_z = tf.contrib.layers.flatten(z)
        dim = flattened_z.get_shape().as_list()[-1]
        assert dim == params["encoding_dim"]

        predictions = {
            "hidden_rep": flattened_z,
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


class Conv3DTupleAE(EvaluateEpochsBaseTF):
    def model_fn(self, features, labels, mode, params):
        encoder_class = params["encoder_class"]
        decoder_class = params["decoder_class"]

        with tf.variable_scope("conv_3d_encoder", reuse=tf.AUTO_REUSE):
            encoder_0 = encoder_class(
                input_key="X_0",
                features=features,
                params=params,
                streamer=self.streamer
            )
        with tf.variable_scope("conv_3d_encoder", reuse=tf.AUTO_REUSE):
            encoder_1 = encoder_class(
                input_key="X_1",
                features=features,
                params=params,
                streamer=self.streamer
            )

        target_0 = encoder_0.get_reconstruction_target()
        target_1 = encoder_1.get_reconstruction_target()
        if params["asymmetric"]:
            target_0 = encoder_1.get_reconstruction_target()
            target_1 = encoder_0.get_reconstruction_target()

        with tf.variable_scope("conv_3d_decoder", reuse=tf.AUTO_REUSE):
            decoder_0 = decoder_class(features, params, encoder_0, target_0)

        with tf.variable_scope("conv_3d_decoder", reuse=tf.AUTO_REUSE):
            decoder_1 = decoder_class(features, params, encoder_1, target_1)

        z_0 = encoder_0.get_encoding()
        y_0 = decoder_0.get_reconstruction()
        x_0 = encoder_0.get_reconstruction_target()

        z_1 = encoder_1.get_encoding()
        y_1 = decoder_1.get_reconstruction()
        x_1 = encoder_1.get_reconstruction_target()

        if params["asymmetric"]:
            tmp = y_0
            y_0 = y_1
            y_1 = tmp

        flattened_z_0 = tf.contrib.layers.flatten(z_0)
        flattened_z_1 = tf.contrib.layers.flatten(z_1)
        dim = flattened_z_0.get_shape().as_list()[-1]

        assert dim == encoder_0.get_encoding_dim()

        predictions = {
            "input": x_0,
            "encoding": flattened_z_0,
            "decoding": y_0,
            "file_name": features["file_name_0"]
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        rec_loss_0 = decoder_0.get_reconstruction_loss()
        rec_loss_1 = decoder_1.get_reconstruction_loss()
        rec_loss = rec_loss_0 / 2 + rec_loss_1 / 2

        # Regularization
        reg_loss = tf.constant(0, dtype=rec_loss.dtype)
        to_reg_0 = z_0
        to_reg_1 = z_1

        diagnose_dim = params["diagnose_dim"]
        hidden_dim = encoder_0.get_encoding_dim()
        if diagnose_dim > 0:
            patient_dim = hidden_dim - diagnose_dim
            patient_encs_0, diag_encs_0 = tf.split(
                z_0,
                [patient_dim, diagnose_dim],
                axis=1
            )

            patient_encs_1, diag_encs_1 = tf.split(
                z_1,
                [patient_dim, diagnose_dim],
                axis=1
            )

            to_reg_0 = diag_encs_0
            to_reg_1 = diag_encs_1

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

        loss = rec_loss + reg_loss

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"]
        )

        train_op = optimizer.minimize(loss, tf.train.get_global_step())

        train_hooks, eval_hooks = self.get_mri_ae_hooks(
            reg_loss=reg_loss,
            rec_loss=rec_loss,
            enc_0=flattened_z_0,
            enc_1=flattened_z_1,
            features=features,
            params=params
        )

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
