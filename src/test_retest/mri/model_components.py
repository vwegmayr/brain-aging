import abc
import tensorflow as tf
from functools import reduce

from src.test_retest import regularizer
from src.test_retest.test_retest_base import \
    linear_trafo_multiple_input_tensors
from src.baum_vagan.tfwrapper import layers


def name_to_hidden_regularization(layer_id, reg_name, activations_test,
                                  activations_retest, weights=None):
    batch_size = tf.shape(activations_test)[0]
    if weights is None:
        weights = tf.ones(shape=[batch_size, 1])
    if reg_name == regularizer.JS_DIVERGENCE_LABEL:
        s_test = tf.nn.softmax(
            activations_test,
            name=str(layer_id) + "_softmax_test"
        )
        s_retest = tf.nn.softmax(
            activations_retest,
            name=str(layer_id) + "_softmax_retest"
        )
        n = activations_test.get_shape().as_list()[1]
        batch_div = regularizer.batch_divergence(
            s_test,
            s_retest,
            n,
            regularizer.js_divergence
        )
        return tf.losses.compute_weighted_loss(batch_div, weights)

    elif reg_name == regularizer.L2_SQUARED_LABEL:
        return tf.losses.mean_squared_error(
                    activations_test, activations_retest,
                    weights=weights
               )
    elif reg_name == regularizer.COSINE_SIMILARITY:
        similarities = regularizer.cosine_similarities(
            activations_test,
            activations_retest
        )
        return tf.losses.compute_weighted_loss(similarities, weights)
    elif reg_name == regularizer.L1_MEAN:
        return regularizer.l1_mean(
            activations_test - activations_retest,
            weights=weights
        )
    else:
        raise ValueError("regularization name '{}' is unknown".format(
                         reg_name))


def normalize_single_image(im):
        mean, var = tf.nn.moments(im, axes=[0])
        std = tf.sqrt(var)
        std += 0.001
        return (im - mean) / std


def normalize_image_batch(batch, voxel_means, voxel_stds):
    # normalize every image
    batch = tf.map_fn(
        normalize_single_image,
        batch
    )

    # voxel normalization computed across train set
    batch = (batch - voxel_means) / voxel_stds

    # normalize every image
    batch = tf.map_fn(
        normalize_single_image,
        batch
    )

    return batch


class Body(abc.ABC):
    def __init__(self, features, params, streamer):
        self.features = features
        self.params = params
        self.streamer = streamer

        self.construct_graph()

    @abc.abstractmethod
    def construct_graph(self):
        pass

    @abc.abstractmethod
    def get_nodes(self):
        pass

    def normalize_voxels(self, x):
        input_shape = self.params["input_shape"]
        input_dim = reduce(lambda a, b: a * b, input_shape)
        voxel_means = tf.constant(
            self.streamer.get_voxel_means(),
            dtype=x.dtype
        )
        voxel_means = tf.reshape(voxel_means, [-1, input_dim])

        voxel_stds = tf.constant(
            self.streamer.get_voxel_stds(),
            dtype=x.dtype
        )
        voxel_stds = tf.reshape(voxel_stds, [-1, input_dim])

        x = tf.reshape(x, [-1, input_dim])
        x = normalize_image_batch(x, voxel_means, voxel_stds)

        return x


class Head(abc.ABC):
    @abc.abstractmethod
    def construct_graph(self):
        pass

    @abc.abstractmethod
    def get_nodes(self):
        pass


class MultiLayerPairEncoder(Body):
    def with_bias(self):
        if "bias" in self.params and self.params["bias"]:
            return True
        else:
            return False

    def non_linear(self):
        if "non_linear" in self.params and self.params["non_linear"]:
            return True
        else:
            return False

    def construct_graph(self):
        features = self.features
        params = self.params

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

            x_0 = normalize_image_batch(x_0, voxel_means, voxel_stds)
            x_1 = normalize_image_batch(x_1, voxel_means, voxel_stds)

        hidden_dim = params["hidden_dim"]
        w = tf.get_variable(
            name="weights",
            shape=[input_dim, hidden_dim],
            dtype=x_0.dtype,
            initializer=tf.contrib.layers.xavier_initializer(seed=43)
        )

        self.x_0 = x_0
        self.x_1 = x_1
        enc_0 = tf.matmul(x_0, w, name="hidden_rep_0")
        enc_1 = tf.matmul(x_1, w, name="hidden_rep_1")

        if self.with_bias():
            bias = tf.get_variable(
                name="bias",
                shape=[1, hidden_dim],
                dtype=enc_0.dtype,
                initializer=tf.initializers.zeros
            )
            enc_0 = tf.add(enc_0, bias)
            enc_1 = tf.add(enc_1, bias)

        if self.non_linear():
            enc_0 = tf.nn.relu(enc_0)
            enc_1 = tf.nn.relu(enc_1)

        self.enc_0 = enc_0
        self.enc_1 = enc_1
        self.w = w

    def get_encoding_weights(self):
        return [self.w]

    def get_encodings(self):
        return self.enc_0, self.enc_1

    def get_nodes(self):
        return self.x_0, self.x_1, self.enc_0, self.enc_1


class MultiLayerPairDecoder(Head):
    def __init__(self, features, params, encoder):
        self.features = features
        self.params = params
        self.encoder = encoder

        self.construct_graph()

    def batch_mean(self, encodings, labels, hc=1):
        comp = 0 * labels + hc
        eq = tf.cast(tf.equal(labels, comp), encodings.dtype)
        batch_mean = tf.reduce_mean(eq * encodings, axis=0)

        return batch_mean

    def construct_graph(self):
        params = self.params
        features = self.features
        x_0, x_1, enc_0, enc_1 = self.encoder.get_nodes()
        w = self.encoder.get_encoding_weights()[0]

        self.rec_0 = tf.matmul(
            enc_0,
            tf.transpose(w),
            name="reconstruction_0"
        )

        self.rec_1 = tf.matmul(
            enc_1,
            tf.transpose(w),
            name="reconstruction_1"
        )

        # Reconstruction
        if not params["asymmetric"]:
            loss_0 = tf.losses.mean_squared_error(x_0, self.rec_0)
            loss_1 = tf.losses.mean_squared_error(x_1, self.rec_1)
        else:
            loss_0 = tf.losses.mean_squared_error(x_0, self.rec_1)
            loss_1 = tf.losses.mean_squared_error(x_1, self.rec_0)
        self.reconstruction_loss = loss_0 / 2 + loss_1 / 2

        # Regularization
        to_reg_0 = enc_0
        to_reg_1 = enc_1
        diagnose_dim = params["diagnose_dim"]
        hidden_dim = params["hidden_dim"]
        if diagnose_dim > 0:
            patient_dim = hidden_dim - diagnose_dim
            patient_encs_0, diag_encs_0 = tf.split(
                enc_0,
                [patient_dim, diagnose_dim],
                axis=1
            )

            patient_encs_1, diag_encs_1 = tf.split(
                enc_1,
                [patient_dim, diagnose_dim],
                axis=1
            )

            to_reg_0 = diag_encs_0
            to_reg_1 = diag_encs_1

        if params["global_diag_encoding"] and diagnose_dim > 0:
            global_diag_ad = tf.get_variable(
                name="global_diag_ad",
                shape=[1, diagnose_dim],
                trainable=False,
                initializer=tf.initializers.zeros
            )

            global_diag_hc = tf.get_variable(
                name="global_diag_hc",
                shape=[1, diagnose_dim],
                trainable=False,
                initializer=tf.initializers.zeros
            )

            labels_0 = features["healthy_0"]
            ad_batch_0 = self.batch_mean(diag_encs_0, labels_0, 0)
            ad_batch_1 = self.batch_mean(diag_encs_1, labels_0, 0)
            ad_batch = ad_batch_0 / 2 + ad_batch_1 / 2

            hc_batch_0 = self.batch_mean(diag_encs_0, labels_0, 1)
            hc_batch_1 = self.batch_mean(diag_encs_1, labels_0, 1)
            hc_batch = hc_batch_0 / 2 + hc_batch_1 / 2

            global_diag_ad = 0.9 * global_diag_ad + 0.1 * ad_batch
            global_diag_hc = 0.9 * global_diag_hc + 0.1 * hc_batch

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

        self.reg_loss = tf.cast(reg_loss, self.reconstruction_loss.dtype)
        self.loss = self.reconstruction_loss + self.reg_loss

    def get_reconstruction_loss(self):
        return self.reconstruction_loss

    def get_regularization_loss(self):
        return self.reg_loss

    def get_total_loss(self):
        return self.loss

    def get_nodes(self):
        return self.rec_0, self.rec_1


class PairClassificationHead(Head):
    def __init__(self, features, params, encodings):
        self.features = features
        self.params = params
        self.encodings = encodings

        self.construct_graph()

    def get_encodings(self):
        return self.encodings

    def get_similarity_weight(self):
        return self.params["similarity_w"]

    def get_dissimilarity_weight(self):
        return self.params["dissimilarity_w"]

    def compute_pair_weighing(self, sim_w, dissim_w, labels_0, labels_1):
        eq = tf.cast(tf.equal(labels_0, labels_1), tf.float32)
        weights = (sim_w + dissim_w) * eq - dissim_w
        return tf.reshape(weights, [-1, 1])

    def construct_graph(self):
        params = self.params
        features = self.features

        enc_0, enc_1 = self.get_encodings()
        # Extract labels
        key = params["target_label_key"]
        self.labels_0 = tf.reshape(features[key + "_0"], [-1])
        self.labels_1 = tf.reshape(features[key + "_1"], [-1])

        # Compute logits
        w = tf.get_variable(
            shape=[enc_0.get_shape()[1], params["n_classes"]],
            dtype=enc_0.dtype,
            name="clf_weight",
            initializer=tf.contrib.layers.xavier_initializer(seed=40)
        )
        b = tf.get_variable(
            shape=[1, params["n_classes"]],
            dtype=enc_0.dtype,
            name="clf_bias",
            initializer=tf.initializers.zeros
        )
        out = [
            tf.add(
                tf.matmul(enc_0, w),
                b
            ),
            tf.add(
                tf.matmul(enc_1, w),
                b
            )
        ]

        self.logits_0, self.logits_1 = out

        self.probs_0 = tf.nn.softmax(self.logits_0)
        self.probs_1 = tf.nn.softmax(self.logits_1)

        self.preds_0 = tf.argmax(input=self.logits_0, axis=1)
        self.preds_1 = tf.argmax(input=self.logits_1, axis=1)

        self.loss_0 = tf.losses.sparse_softmax_cross_entropy(
            labels=self.labels_0,
            logits=self.logits_0
        )

        self.loss_1 = tf.losses.sparse_softmax_cross_entropy(
            labels=self.labels_1,
            logits=self.logits_1
        )

        self.loss_clf = self.loss_0 / 2 + self.loss_1 / 2

        self.acc_0 = tf.reduce_mean(
            tf.cast(tf.equal(self.preds_0, self.labels_0), tf.float32)
        )

        self.acc_1 = tf.reduce_mean(
            tf.cast(tf.equal(self.preds_1, self.labels_1), tf.float32)
        )

        # Set some regularizers
        self.loss_o = tf.constant(0, dtype=self.loss_clf.dtype)
        if params["lambda_o"] != 0:
            self.output_regularization()

        self.loss_h = tf.constant(0, dtype=self.loss_clf.dtype)
        if params["hidden_lambda"] != 0:
            self.hidden_regularization()

    def output_regularization(self):
        params = self.params
        lam = params["lambda_o"]
        key = "output_regularizer"

        label_equality = tf.cast(
            tf.equal(self.labels_0, self.labels_1),
            tf.float32
        )
        if params[key] == "kl_divergence":
            loss_o = regularizer.batch_divergence(
                self.probs_0,
                self.probs_1,
                params["n_classes"],
                regularizer.kl_divergence
            )
        elif params[key] == "js_divergence":
            loss_o = regularizer.batch_divergence(
                self.probs_0,
                self.probs_1,
                params["n_classes"],
                regularizer.js_divergence
            )
        else:
            raise ValueError("Regularizer not found")

        weights = tf.reshape(label_equality, [-1, 1])
        loss_o = tf.losses.compute_weighted_loss(loss_o, weights)
        self.loss_o = lam * loss_o

    def hidden_regularization(self):
        enc_0, enc_1 = self.get_encodings()
        params = self.params
        diagnose_dim = params["diagnose_dim"]
        enc_dim = enc_0.get_shape().as_list()[-1]
        to_reg_0 = enc_0
        to_reg_1 = enc_1

        if diagnose_dim > 0:
            patient_dim = enc_dim - diagnose_dim
            patient_encs_0, diag_encs_0 = tf.split(
                enc_0,
                [patient_dim, diagnose_dim],
                axis=1
            )

            patient_encs_1, diag_encs_1 = tf.split(
                enc_1,
                [patient_dim, diagnose_dim],
                axis=1
            )

            to_reg_0 = diag_encs_0
            to_reg_1 = diag_encs_1

        weights = self.compute_pair_weighing(
            sim_w=self.get_similarity_weight(),
            dissim_w=self.get_dissimilarity_weight(),
            labels_0=self.labels_0,
            labels_1=self.labels_1
        )

        reg = name_to_hidden_regularization(
            "hidden_reg",
            params["hidden_regularizer"],
            to_reg_0,
            to_reg_1,
            weights
        )

        self.loss_h = params["hidden_lambda"] * reg

    def get_losses_with_names(self):
        ops = [self.loss_o, self.loss_h, self.loss_clf]
        names = ["output_loss", "hidden_loss", "cross_entropy"]
        return ops, names

    def get_acc_test(self):
        return self.acc_0

    def get_accuracy(self):
        return self.acc_0 / 2 + self.acc_1 / 2

    def get_total_loss(self):
        return self.loss_clf + self.loss_o + self.loss_h

    def get_predictions(self):
        return self.preds_0, self.preds_1

    def get_nodes(self):
        return self.get_predictions(), self.get_total_loss()


class Conv3DEncoder(Body):
    def __init__(self, input_key, *args, **kwargs):
        self.input_key = input_key
        super(Conv3DEncoder, self).__init__(
            *args,
            **kwargs
        )

    def get_encoding_dim(self):
        return self.params["hidden_dim"]

    def get_pooling_size(self, i):
        return self.params["pooling_sizes"][i]

    def use_pooling(self):
        return "pooling_sizes" in self.params

    def construct_graph(self):
        features = self.features
        params = self.params

        x = features[self.input_key]
        n_filters = params["n_filters"]
        filter_sizes = params["filter_sizes"]

        if params["normalize_images"]:
            x = self.normalize_voxels(x)

        input_shape = params["input_shape"]
        # Reshape to have one explicit channel
        x = tf.reshape(
            x,
            [-1, input_shape[0], input_shape[1], input_shape[2], 1]
        )

        self.x = x
        current_input = x

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
                        current_input, W, strides=[1, 2, 2, 2, 1],
                        padding='SAME'
                    ),
                    b
                )
            )
            current_input = output

            # Pooling
            if self.use_pooling():
                s = self.get_pooling_size(layer_i)
                current_input = tf.layers.max_pooling3d(
                    inputs=current_input,
                    pool_size=s,
                    strides=s
                )
            dim_list = current_input.get_shape().as_list()[1:]
            cur_dim = reduce(lambda x, y: x * y, dim_list)
            print(">>>>> input dim is {}".format(cur_dim))

        self.z = current_input

        dim_list = current_input.get_shape().as_list()[1:]
        cur_dim = reduce(lambda x, y: x * y, dim_list)
        print(">>>>> input dim after conv layers: {}".format(cur_dim))

        self.linear_trafo = False
        if cur_dim > self.get_encoding_dim() and not self.use_pooling():
            print("Non conv layer needed")
            self.linear_trafo = True
            self.dim_before_linear_trafo = cur_dim
            self.dim_list = dim_list
            current_input = tf.contrib.layers.flatten(current_input)
            W = tf.get_variable(
                "non_conv_w",
                shape=[cur_dim, self.get_encoding_dim()],
                initializer=tf.contrib.layers.xavier_initializer(seed=40)
            )
            b = tf.get_variable(
                "non_conv_b",
                shape=[1, self.get_encoding_dim()],
                initializer=tf.initializers.zeros
            )
            encoder.append(W)

            current_input = tf.add(
                tf.nn.relu(tf.matmul(current_input, W)),
                b
            )

            self.z = current_input
        else:
            self.z = tf.contrib.layers.flatten(current_input)

        self.encoder_weights = encoder
        self.encoder_shapes = shapes

    def get_encoding(self):
        return self.z

    def get_reconstruction_target(self):
        return self.x

    def get_nodes(self):
        return self.x, self.z


class Conv3DDecoder(Head):
    def __init__(self, features, params, encoder, target):
        self.features = features
        self.params = params
        self.encoder = encoder
        self.target = target

        self.construct_graph()

    def construct_graph(self):
        params = self.params

        x = self.target
        z = self.encoder.get_encoding()
        encoder_weights = self.encoder.encoder_weights
        encoder_shapes = self.encoder.encoder_shapes

        current_input = z
        # Check if the encoder performed a final non-convolutional
        # transformation.
        if self.encoder.linear_trafo:
            dim = self.encoder.dim_before_linear_trafo
            if not params["tied_weights"]:
                W = tf.get_variable(
                    "non_conv_w_dec",
                    shape=[dim, params[self.encoder.get_encoding_dim()]],
                    initializer=tf.contrib.layers.xavier_initializer(seed=40)
                )
            else:
                W = encoder_weights[-1]

            b = tf.get_variable(
                "non_conv_b_dec",
                shape=[1, dim],
                initializer=tf.initializers.zeros
            )

            current_input = tf.add(
                tf.nn.relu(tf.matmul(current_input, tf.transpose(W))),
                b
            )
            # Unflatten
            dim_list = self.encoder.dim_list
            current_input = tf.reshape(current_input, [-1] + dim_list)
            encoder_weights = encoder_weights[:-1]

        encoder_weights.reverse()
        encoder_shapes.reverse()
        # Build the decoder
        for layer_i, shape in enumerate(encoder_shapes):
            # deconv
            W = encoder_weights[layer_i]
            b = tf.get_variable(
                name="bias_deconv_layer_" + str(layer_i),
                shape=W.get_shape().as_list()[3],
                initializer=tf.initializers.zeros
            )

            output = tf.nn.relu(
                tf.add(
                    tf.nn.conv3d_transpose(
                        current_input, W,
                        tf.stack([tf.shape(x)[0], shape[1], shape[2],
                                  shape[3], shape[4]]),
                        strides=[1, 2, 2, 2, 1], padding='SAME'
                    ),
                    b
                )
            )
            current_input = output

        self.y = current_input
        self.reconstruction_loss = tf.losses.mean_squared_error(x, self.y)

    def get_reconstruction_loss(self):
        return self.reconstruction_loss

    def get_reconstruction(self):
        return self.y

    def get_nodes(self):
        return [self.y]


class Conv3DUnetEncoder(Body):
    def __init__(self, input_key, *args, **kwargs):
        self.input_key = input_key
        super(Conv3DUnetEncoder, self).__init__(
            *args,
            **kwargs
        )

    def get_encoding_dim(self):
        return self.params["hidden_dim"]

    def get_pooling_size(self, i):
        return self.params["pooling_sizes"][i]

    def use_pooling(self):
        return "pooling_sizes" in self.params

    def construct_graph(self):
        features = self.features
        params = self.params

        x = features[self.input_key]
        n_filters = params["n_filters"]
        filter_sizes = params["filter_sizes"]

        if params["normalize_images"]:
            x = self.normalize_voxels(x)

        input_shape = params["input_shape"]
        # Reshape to have one explicit channel
        x = tf.reshape(
            x,
            [-1, input_shape[0], input_shape[1], input_shape[2], 1]
        )

        self.x = x
        current_input = x

        n_ch_0 = 16
        conv1_1 = layers.conv3D_layer(x, 'conv1_1', num_filters=n_ch_0)
        conv1_2 = layers.conv3D_layer(conv1_1, 'conv1_2', num_filters=n_ch_0)
        pool1 = layers.maxpool3D_layer(conv1_2)

        conv2_1 = layers.conv3D_layer(pool1, 'conv2_1', num_filters=n_ch_0*2)
        conv2_2 = layers.conv3D_layer(conv2_1, 'conv2_2', num_filters=n_ch_0*2)
        pool2 = layers.maxpool3D_layer(conv2_2)

        conv3_1 = layers.conv3D_layer(pool2, 'conv3_1', num_filters=n_ch_0*4)
        conv3_2 = layers.conv3D_layer(conv3_1, 'conv3_2', num_filters=n_ch_0*4)
        # pool3 = layers.maxpool3D_layer(conv3_2)

        # conv4_1 = layers.conv3D_layer(pool3, 'conv4_1', num_filters=n_ch_0*8)
        # conv4_2 = layers.conv3D_layer(conv4_1, 'conv4_2', num_filters=n_ch_0*8)

        self.z = tf.contrib.layers.flatten(conv3_2)
        current_input = self.z

        dim_list = current_input.get_shape().as_list()[1:]
        cur_dim = reduce(lambda x, y: x * y, dim_list)
        self.linear_trafo = False
        if cur_dim > self.get_encoding_dim():
            print("Non conv layer needed")
            self.linear_trafo = True
            self.dim_before_linear_trafo = cur_dim
            self.dim_list = dim_list
            current_input = tf.contrib.layers.flatten(current_input)
            W = tf.get_variable(
                "non_conv_w",
                shape=[cur_dim, self.get_encoding_dim()],
                initializer=tf.contrib.layers.xavier_initializer(seed=40)
            )
            b = tf.get_variable(
                "non_conv_b",
                shape=[1, self.get_encoding_dim()],
                initializer=tf.initializers.zeros
            )

            self.linear_w = W

            current_input = tf.add(
                tf.nn.relu(tf.matmul(current_input, W)),
                b
            )

            self.z = current_input

        self.conv1_1 = conv1_1
        self.conv1_2 = conv1_2
        self.pool1 = pool1
        self.conv2_1 = conv2_1
        self.conv2_2 = conv2_2
        self.pool2 = pool2
        self.conv3_1 = conv3_1
        self.conv3_2 = conv3_2
        # self.pool3 = pool3
        # self.conv4_1 = conv4_1
        # self.conv4_2 = conv4_2

    def get_encoding(self):
        return self.z

    def get_reconstruction_target(self):
        return self.x

    def get_nodes(self):
        return self.x, self.z


class Conv3DUnetDecoder(Head):
    def __init__(self, features, params, encoder, target):
        self.features = features
        self.params = params
        self.encoder = encoder
        self.target = target

        self.construct_graph()

    def construct_graph(self):
        x = self.target

        n_ch_0 = 16
        # conv4_2 = self.encoder.conv4_2
        conv3_2 = self.encoder.conv3_2
        conv2_2 = self.encoder.conv2_2
        conv1_2 = self.encoder.conv1_2

        current_input = conv3_2
        if self.encoder.linear_trafo:
            dim = self.encoder.dim_before_linear_trafo

            W = self.encoder.linear_w

            b = tf.get_variable(
                "non_conv_b_dec",
                shape=[1, dim],
                initializer=tf.initializers.zeros
            )

            current_input = tf.add(
                tf.nn.relu(tf.matmul(self.encoder.z, tf.transpose(W))),
                b
            )
            current_input = tf.reshape(current_input, [-1] + conv3_2.get_shape().as_list()[1:])

        """
        upconv3 = layers.deconv3D_layer(current_input, name='upconv3', num_filters=n_ch_0)
        concat3 = layers.crop_and_concat_layer_fixed([upconv3, conv3_2], axis=-1)

        conv5_1 = layers.conv3D_layer(concat3, 'conv5_1', num_filters=n_ch_0*4)

        conv5_2 = layers.conv3D_layer(conv5_1, 'conv5_2', num_filters=n_ch_0*4)
        """

        upconv2 = layers.deconv3D_layer(current_input, name='upconv2', num_filters=n_ch_0)
        concat2 = layers.crop_and_concat_layer_fixed([upconv2, conv2_2], axis=-1)

        conv6_1 = layers.conv3D_layer(concat2, 'conv6_1', num_filters=n_ch_0*2)
        conv6_2 = layers.conv3D_layer(conv6_1, 'conv6_2', num_filters=n_ch_0*2)

        upconv1 = layers.deconv3D_layer(conv6_2, name='upconv1', num_filters=n_ch_0)
        concat1 = layers.crop_and_concat_layer_fixed([upconv1, conv1_2], axis=-1)

        conv8_1 = layers.conv3D_layer(concat1, 'conv8_1', num_filters=n_ch_0)
        conv8_2 = layers.conv3D_layer(conv8_1, 'conv8_2', num_filters=1, activation=tf.identity)

        self.y = conv8_2
        self.reconstruction_loss = tf.losses.mean_squared_error(x, self.y)

    def get_reconstruction_loss(self):
        return self.reconstruction_loss

    def get_reconstruction(self):
        return self.y

    def get_nodes(self):
        return [self.y] 


def single_classification_head(x, y, class_weights, n_classes, scope_name="clf_head", scope_reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        logits = tf.layers.dense(x, n_classes)
        y = tf.reshape(y, [-1])

        probs = tf.nn.softmax(logits)
        preds = tf.argmax(input=logits, axis=1)

        class_0_w = class_weights[0]
        class_1_w = class_weights[1]
        weights = class_1_w * tf.cast(y, tf.float32) + class_0_w * (1 - tf.cast(y, tf.float32)) 

        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=y,
            logits=logits,
            weights=weights
        )

        acc = tf.reduce_mean(
            tf.cast(tf.equal(preds, y), tf.float32)
        )

        return probs, preds, loss, acc


def C3D_fcn_16_body(x, training, scope_name='classifier', scope_reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer(x, 'conv1_1', num_filters=16)

        pool1 = layers.maxpool3D_layer(conv1_1)

        conv2_1 = layers.conv3D_layer(pool1, 'conv2_1', num_filters=32)

        pool2 = layers.maxpool3D_layer(conv2_1)

        conv3_1 = layers.conv3D_layer(pool2, 'conv3_1', num_filters=64)
        conv3_2 = layers.conv3D_layer(conv3_1, 'conv3_2', num_filters=64)

        pool3 = layers.maxpool3D_layer(conv3_2)

        conv4_1 = layers.conv3D_layer(pool3, 'conv4_1', num_filters=128)
        conv4_2 = layers.conv3D_layer(conv4_1, 'conv4_2', num_filters=128)

        pool4 = layers.maxpool3D_layer(conv4_2)

        conv5_1 = layers.conv3D_layer(pool4, 'conv5_1', num_filters=256)
        conv5_2 = layers.conv3D_layer(conv5_1, 'conv5_2', num_filters=256)

        convD_1 = layers.conv3D_layer(conv5_2, 'convD_1', num_filters=256)

        logits = tf.reduce_mean(convD_1, axis=(1, 2, 3))

    return logits


def C3D_fcn_16_bn_body(x, training, scope_name='classifier', scope_reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv3D_layer_bn(x, 'conv1_1', training, num_filters=16)

        pool1 = layers.maxpool3D_layer(conv1_1)

        conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', training, num_filters=32)

        pool2 = layers.maxpool3D_layer(conv2_1)

        conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', training, num_filters=64)
        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', training, num_filters=64)

        pool3 = layers.maxpool3D_layer(conv3_2)

        conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', training, num_filters=128)
        conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', training, num_filters=128)

        pool4 = layers.maxpool3D_layer(conv4_2)

        conv5_1 = layers.conv3D_layer_bn(pool4, 'conv5_1', training, num_filters=256)
        conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', training, num_filters=256)

        convD_1 = layers.conv3D_layer_bn(conv5_2, 'convD_1', training, num_filters=256)

        logits = tf.reduce_mean(convD_1, axis=(1, 2, 3))

    return logits


def C2D_fcn_16_bn_body(x, training, scope_name='classifier', scope_reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv2D_layer_bn(x, 'conv1_1', training, num_filters=16)

        pool1 = layers.maxpool2D_layer(conv1_1)

        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', training, num_filters=32)

        pool2 = layers.maxpool2D_layer(conv2_1)

        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', training, num_filters=64)
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', training, num_filters=64)

        pool3 = layers.maxpool2D_layer(conv3_2)

        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', training, num_filters=128)
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', training, num_filters=128)

        pool4 = layers.maxpool2D_layer(conv4_2)

        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', training, num_filters=256)
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', training, num_filters=256)

        convD_1 = layers.conv2D_layer_bn(conv5_2, 'convD_1', training, num_filters=256)

        logits = tf.reduce_mean(convD_1, axis=(1, 2))

    return logits


def C2D_fcn_16_body(x, training, scope_name='classifier', scope_reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv2D_layer(x, 'conv1_1', num_filters=16)

        pool1 = layers.maxpool2D_layer(conv1_1)

        conv2_1 = layers.conv2D_layer(pool1, 'conv2_1', num_filters=32)

        pool2 = layers.maxpool2D_layer(conv2_1)

        conv3_1 = layers.conv2D_layer(pool2, 'conv3_1', num_filters=64)
        conv3_2 = layers.conv2D_layer(conv3_1, 'conv3_2', num_filters=64)

        pool3 = layers.maxpool2D_layer(conv3_2)

        conv4_1 = layers.conv2D_layer(pool3, 'conv4_1', num_filters=128)
        conv4_2 = layers.conv2D_layer(conv4_1, 'conv4_2', num_filters=128)

        pool4 = layers.maxpool2D_layer(conv4_2)

        conv5_1 = layers.conv2D_layer(pool4, 'conv5_1', num_filters=256)
        conv5_2 = layers.conv2D_layer(conv5_1, 'conv5_2', num_filters=256)

        convD_1 = layers.conv2D_layer(conv5_2, 'convD_1', num_filters=256)

        logits = tf.reduce_mean(convD_1, axis=(1, 2))

    return logits
