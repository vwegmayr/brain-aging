import tensorflow as tf
import math
import numpy as np
from modules.models.utils import custom_print


def parse_function(string, modules):
    submodules = string.split('.')
    if submodules[0] not in modules:
        return None
    func = modules[submodules[0]]
    for submodule in submodules[1:]:
        func = getattr(func, submodule)
        assert(func is not None)
    return func


def func_fwd_input(fn):
    return lambda context, input, **kwargs: fn(input, **kwargs)


class DeepNNLayers(object):
    def __init__(self, print_shapes=True):
        self.is_training = True
        self.debug_summaries = False
        self.cnn_layers_shapes = []
        self.enable_print_shapes = print_shapes
        self.parse_layers_defs = {
            'concat_layers': self._parse_concat_layers,
            'conditionnal_branch': self.conditionnal_branch,
            'conv2d': func_fwd_input(self.conv2d_layer),
            'conv3d': func_fwd_input(self.conv3d_layer),
        }
        for f in [
            'batch_norm', 'batch_renorm',
            'normalize_image', 'residual_block', 'localized_batch_norm',
            'dataset_norm_online', 'voxel_wide_norm_online',
            'apply_gaussian', 'conv2d_shared_all_dims_layer',
            'random_crop', 'local_norm_image', 'random_rot',
            'image_summary',
        ]:
            self.parse_layers_defs[f] = func_fwd_input(getattr(self, f))

    # ================= Parsing of CNN architecture =================
    def _parse_concat_layers(self, context, input, layers_def):
        layers_out = []
        for l in layers_def:
            layers_out.append(self.parse_single_layer(context, input, l))
        return tf.concat(layers_out, 4)

    def parse_layers(self, context, input, layers_def):
        assert(isinstance(layers_def, list))
        for l in layers_def:
            input = self.parse_single_layer(context, input, l)
        return input

    def parse_single_layer(self, context, input, layer_def):
        layer_type = layer_def['type']
        if layer_type in self.parse_layers_defs:
            func_ptr = self.parse_layers_defs[layer_type]
        else:
            # Parse function name
            func_ptr = parse_function(layer_type, {
                'tf': tf,
            })
            func_ptr = func_fwd_input(func_ptr)
        params = layer_def.copy()
        del params['type']
        return func_ptr(context, input, **params)

    # ================= Summaries and utils =================
    def print_shape(self, text):
        if self.enable_print_shapes:
            custom_print(text)

    def on_cnn_layer(self, layer, name=None):
        if name is None:
            name = layer.name.split('/')[1]
        self.cnn_layers_shapes.append({
            'shape': layer.get_shape().as_list(),
            'name': name,
        })

    def variable_summaries(self, var, name, fullcontent=True):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('%s_summary' % name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('hist', var)

    def convet_filters_summary(self, w, name):
        """
        input:
        @filters: [x, y, if, of] shaped tensor with
            - @if number of features in input
                Only value '1' is supported
            - @of number of filters
        """
        w_shape = w.get_shape().as_list()
        assert(w_shape[2] in [1, 3])
        with tf.name_scope(name):
            output_filters = w_shape[3]
            num_rows = int(math.ceil(math.sqrt(output_filters)))
            num_cols = int(math.ceil(output_filters/float(num_rows)))

            # Pad W with more filters if needed
            if w_shape[3] != num_rows*num_cols:
                z = tf.zeros(
                    w_shape[0:3] + [num_rows*num_cols - w_shape[3]],
                    dtype=tf.float32,
                )
                w = tf.concat([w, z], 3)
            # Pad x y only
            padding = np.array([
                [2, 2],  # x
                [2, 2],  # y
                [0, 0],  # input filters
                [0, 0],  # output filters
            ])
            w = tf.pad(w, padding, "CONSTANT")
            w_list = tf.split(
                w,
                num_or_size_splits=num_cols*num_rows,
                axis=3,
            )
            rows = [
                tf.concat(w_list[i*num_cols:i*num_cols+num_cols], 0)
                for i in range(num_rows)
            ]
            img = tf.concat(rows, 1)  # [x, y, if, 1]
            tf.summary.image(
                name,
                tf.reshape(img, [1] + img.get_shape().as_list()[0:3]),
            )

    def image_summary(self, x, name="image_summary"):
        with tf.name_scope(name):
            tf.summary.image(
                'image',
                x[:, :, :, int(x.get_shape().as_list()[3]/2), 0:1],
            )
        return x

    # ================= Generic ConvNets =================
    def conv_layer_wrapper(
        self,
        func,
        x,
        num_filters,
        filter_weights=[3, 3, 3],
        nl=tf.nn.relu,
        strides=[2, 2, 2],
        padding='SAME',
        name="conv3d_layer",
        conv_type='conv',
        bn=True,
        reversed_filters=False,
    ):
        assert(conv_type in ['conv', 'deconv'])
        with tf.variable_scope(name):
            conv_input_shape = x.get_shape()[1:].as_list()
            input_channels = conv_input_shape[-1]
            if conv_type == 'conv':
                W_shape = filter_weights + [input_channels, num_filters]
            else:
                W_shape = filter_weights + [num_filters, input_channels]
            W = tf.get_variable(
                "w",
                shape=[np.prod(filter_weights)] +
                [input_channels, num_filters],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l1_regularizer(1.0),
            )
            W = tf.reshape(W, W_shape)
            if reversed_filters:
                additionnal_filters = [
                    W,
                    tf.reverse(W, [0]),
                    tf.reverse(W, [1]),
                    tf.reverse(W, [0, 1]),
                ]
                if len(filter_weights) == 3:
                    additionnal_filters += [
                        tf.reverse(W, [2]),
                        tf.reverse(W, [0, 2]),
                        tf.reverse(W, [1, 2]),
                        tf.reverse(W, [0, 1, 2]),
                    ]
                W = tf.concat(additionnal_filters, -1)
            out = func(
                x,
                filter=W,
                strides=[1] + strides + [1],
                padding=padding,
            )
            if bn:
                out = self.batch_norm(out)
            else:
                b = tf.get_variable(
                    "b",
                    [num_filters],
                    initializer=tf.constant_initializer(0.001),
                )
                if reversed_filters:
                    b = tf.concat([b] * len(additionnal_filters), -1)
                out += b
            out = nl(out)
            if self.debug_summaries:
                self.variable_summaries(W, "w")
                self.variable_summaries(out, "output")
            self.print_shape('%s -> [%s] -> %s' % (
                conv_input_shape,
                tf.contrib.framework.get_name_scope(),
                out.get_shape()[1:].as_list()
            ))
        return out

    # ================= 2D ConvNets =================
    def conv2d_layer(self, *args, **kwargs):
        # Some default values are for 3d convnets, so fix it
        if 'filter_weights' not in kwargs:
            kwargs['filter_weights'] = [3, 3]
        if 'strides' not in kwargs:
            kwargs['strides'] = [2, 2]
        return self.conv_layer_wrapper(tf.nn.conv2d, *args, **kwargs)

    def conv2d_layer_transpose(self, output_shape, *args, **kwargs):
        # Some default values are for 3d convnets, so fix it
        if 'filter_weights' not in kwargs:
            kwargs['filter_weights'] = [3, 3]
        if 'strides' not in kwargs:
            kwargs['strides'] = [2, 2]
        kwargs['conv_type'] = 'deconv'
        assert(len(output_shape) == 4)

        def conv2d_transpose_func(input, **kwargs2):
            # Dynamic batch_size
            _output_shape = output_shape[1:]
            _output_shape = [tf.shape(input)[0]] + _output_shape
            return tf.nn.conv2d_transpose(
                input,
                output_shape=_output_shape,
                **kwargs2
            )
        return self.conv_layer_wrapper(
            conv2d_transpose_func,
            *args,
            **kwargs
        )

    # ================= 3D ConvNets =================
    def conv2d_shared_all_dims_layer(
        self,
        _input,
        name,
        s=5,
        num_filters_per_dim=8,
        *args,
        **kwargs
    ):
        def do_c(filters, *args, **kwargs):
            return self.conv3d_layer(
                _input,
                num_filters=num_filters_per_dim,
                filter_weights=filters,
                name='conv',
                padding='SAME',
                *args,
                **kwargs
            )
        with tf.variable_scope(name) as tf_scope:
            b1 = do_c([s, s, 1], *args, **kwargs)
            tf_scope.reuse_variables()
            b2 = do_c([s, 1, s], *args, **kwargs)
            b3 = do_c([1, s, s], *args, **kwargs)
            return tf.concat([b1, b2, b3], 4)

    def conv3d_layer(self, *args, **kwargs):
        return self.conv_layer_wrapper(tf.nn.conv3d, *args, **kwargs)

    def conv3d_layer_transpose(self, output_shape, *args, **kwargs):

        def conv3d_transpose_func(input, **kwargs2):
            # Dynamic batch_size
            _output_shape = output_shape[1:]
            _output_shape = [tf.shape(input)[0]] + _output_shape
            return tf.nn.conv3d_transpose(
                input,
                output_shape=_output_shape,
                **kwargs2
            )
        assert(len(output_shape) == 5)
        kwargs['conv_type'] = 'deconv'
        return self.conv_layer_wrapper(
            conv3d_transpose_func,
            *args,
            **kwargs
        )

    # ================= Other layers =================
    def fc_layer(self, x, num_outputs, nl=tf.nn.relu, name="unnamedfc"):
        with tf.variable_scope(name):
            num_inputs = x.get_shape()[1:].as_list()[0]
            W_fc = tf.get_variable(
                "w",
                shape=[num_inputs, num_outputs],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l1_regularizer(1.0),
            )
            b_fc = tf.get_variable(
                "b",
                [num_outputs],
                initializer=tf.constant_initializer(0.1),
            )
            out = nl(tf.matmul(x, W_fc) + b_fc)
            if self.debug_summaries:
                self.variable_summaries(W_fc, "W")
                self.variable_summaries(b_fc, "b")
                self.variable_summaries(out, "output")
            self.print_shape('%s -> [%s] -> %s' % (
                x.get_shape().as_list()[1:],
                tf.contrib.framework.get_name_scope(),
                out.get_shape().as_list()[1:],
            ))
        return out


    def residual_block(self, x, name, num_features=None):
        with tf.variable_scope(name):
            if num_features is None:
                num_features = x.get_shape().as_list()[-1]
                assert(num_features is not None)
            shortcut = x
            x = self.conv3d_layer(x, num_features, strides=[1, 1, 1], bn=True, nl=tf.nn.relu, name='conv1')
            x = self.conv3d_layer(x, num_features, strides=[1, 1, 1], bn=True, nl=tf.identity, name='conv2')
            x += shortcut
            return tf.nn.relu(x)

    def conditionnal_branch(self, context, x, global_step_threshold, before_threshold, after_threshold):
        output_before_threshold = self.parse_layers(context, x, before_threshold)
        output_after_threshold = self.parse_layers(context, x, after_threshold)
        return tf.cond(
            tf.train.get_global_step() < global_step_threshold,
            lambda: output_before_threshold,
            lambda: output_after_threshold,
        )

    # ================= Regularization =================
    def batch_norm(self, x, decay=0.9, **kwargs):
        if 'training' not in kwargs:
            kwargs['training'] = self.is_training
        return tf.layers.batch_normalization(
            x,
            momentum=decay,
            **kwargs
        )

    def batch_renorm(
        self,
        x,
        decay=0.9,
        transition_begin_iter=250,
        transition_d_end_iter=850,
        transition_r_end_iter=1500,
        max_r=3,
        max_d=5,
        **kwargs
    ):
        max_r = tf.cast(max_r, tf.float32)
        max_d = tf.cast(max_d, tf.float32)
        step = tf.cast(tf.train.get_global_step(), tf.float32)
        r = (step - transition_begin_iter) * max_r / \
            (transition_r_end_iter - transition_begin_iter)
        r += (step - transition_r_end_iter) * 1.0 / \
            (transition_begin_iter - transition_r_end_iter)
        r = tf.clip_by_value(r, 1.0, max_r)
        d = (step - transition_begin_iter) * max_d / \
            (transition_d_end_iter - transition_begin_iter)
        d = tf.clip_by_value(d, 0.0, max_d)
        renorm_clipping = {
            'rmin': 1./r,
            'rmax': r,
            'dmax': d,
        }
        return tf.layers.batch_normalization(
            x,
            training=self.is_training,
            momentum=decay,
            renorm=True,
            renorm_clipping=renorm_clipping,
            **kwargs
        )

    def apply_gaussian(self, x, sigma, kernel_half_size=None):
        if kernel_half_size is None:
            kernel_half_size = int(4 * sigma)
        dim_size = kernel_half_size * 2 + 1
        fx, fy, fz = np.mgrid[
            -kernel_half_size:kernel_half_size+1,
            -kernel_half_size:kernel_half_size+1,
            -kernel_half_size:kernel_half_size+1,
        ]
        gauss_filter = np.exp(-(fx**2 + fy**2 + fz**2)/float(sigma*sigma*2))
        gauss_filter = gauss_filter / gauss_filter.sum()
        gauss_filter = gauss_filter.reshape(
            [dim_size, dim_size, dim_size, 1, 1],
        )
        return tf.nn.conv3d(
            x,
            gauss_filter,
            strides=[1, 1, 1, 1, 1],
            padding='SAME',
        )

    def dropout(self, x, prob):
        if not self.is_training:
            return x
        return tf.nn.dropout(x, keep_prob=prob)

    def random_crop(self, x, new_shape=None, new_shape_ratio=None):
        if new_shape is not None:
            assert(new_shape_ratio is None)
        elif new_shape_ratio is not None:
            new_shape = x.get_shape().as_list()[1:]
            for i, ratio in enumerate(new_shape_ratio):
                new_shape[i] = int(new_shape[i] * ratio)
        else:
            assert(False)
        new_shape = tf.convert_to_tensor(new_shape)
        new_shape = tf.concat([[tf.shape(x)[0]], new_shape], 0)

        out = tf.random_crop(
            x,
            new_shape,
        )
        self.print_shape('%s -> [random_crop] -> %s' % (
            x.get_shape().as_list()[1:],
            out.get_shape().as_list()[1:],
        ))
        return out

    def random_rot(self, x, max_angle=0.2, axis=0):
        batch_size = tf.shape(x)[0]
        angles = tf.random_uniform(
            [batch_size],
            minval=-max_angle,
            maxval=+max_angle,
            dtype=tf.float32,
        )
        tf.summary.histogram('random_rot/angles', angles)

        def rotate_single_image(angle, img):
            result = tf.contrib.image.rotate(
                img,
                angle,
                interpolation='BILINEAR',
                name='rotate_single_image',
            )
            return tf.expand_dims(result, 0)

        def body(i, r):
            return [
                tf.add(i, 1),
                tf.concat([r, rotate_single_image(angles[i], x[i])], 0),
            ]
        assert(axis in [0, 1, 2])
        axis += 1
        perm = [0, 1, 2, 3, 4]
        perm[1] = axis
        perm[axis] = 1
        x = tf.transpose(x, perm)
        i = tf.constant(1)
        result = rotate_single_image(angles[0], x[0])
        r = tf.while_loop(
            lambda i, _: tf.less(i, batch_size),
            body,
            [i, result],
            shape_invariants=[
                i.get_shape(),
                tf.TensorShape([None] + x.get_shape().as_list()[1:]),
            ]
        )
        return tf.transpose(r[1], perm)

    # ================= Images Normalization =================
    def localized_batch_norm(self, x, kernel_half_size, sigma, eps=0.001):
        smoothed_x = self.apply_gaussian(
            x,
            kernel_half_size=kernel_half_size,
            sigma=sigma,
        )
        smoothed_x2 = self.apply_gaussian(
            x ** 2,
            kernel_half_size=kernel_half_size,
            sigma=sigma,
        )
        smoothed_x = tf.reduce_mean(smoothed_x, axis=0, keep_dims=True)
        smoothed_x2 = tf.reduce_mean(smoothed_x2, axis=0, keep_dims=True)
        variance = smoothed_x2 - smoothed_x * smoothed_x
        return (x - smoothed_x) / (tf.sqrt(variance + eps))

    def voxel_wide_norm_online(
        self,
        x,
        compute_moments_on_smoothed=None,
    ):
        with tf.variable_scope('voxel_wide_norm_online'):
            image_shape = x.get_shape()[1:]
            accumulated_count = tf.Variable(
                0,
                dtype=tf.float32,
                trainable=False,
                name='accumulated_count',
            )
            accumulated_x = tf.Variable(
                np.zeros(image_shape, np.float32),
                dtype=tf.float32,
                trainable=False,
                name='accumulated_x',
            )
            accumulated_x2 = tf.Variable(
                np.zeros(image_shape, np.float32),
                dtype=tf.float32,
                trainable=False,
                name='accumulated_x2',
            )
            x_for_computations = x
            if compute_moments_on_smoothed is not None:
                x_for_computations = self.apply_gaussian(
                    x,
                    **compute_moments_on_smoothed
                )
            x_mean = tf.reduce_mean(
                x_for_computations,
                axis=[0],
                keep_dims=False,
            )
            x2_mean = tf.reduce_mean(
                x_for_computations ** 2,
                axis=[0],
                keep_dims=False,
            )
            if self.is_training:
                accumulated_count = accumulated_count.assign_add(1.0)
                accumulated_x = accumulated_x.assign_add(x_mean),
                accumulated_x2 = accumulated_x2.assign_add(x2_mean)
            mean = accumulated_x / accumulated_count
            variance = accumulated_x2 / accumulated_count
            variance -= mean ** 2
            return (x - mean) / tf.sqrt(variance + 0.001)

    def dataset_norm_online(self, x, **kwargs):
        with tf.variable_scope('dataset_norm_online'):
            # 1. Image normalization
            x = self.normalize_image(x)
            # 2. Voxel normalization
            x = self.voxel_wide_norm_online(x, **kwargs)
            # 3. Image normalization again
            return self.normalize_image(x)

    def normalize_image(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2, 3, 4], keep_dims=True)
        return (x - mean) / tf.sqrt(var + 0.0001)

    def local_norm_image(self, x, gaussian_params={}, div_eps=1.):
        # Increase 'div_eps' to hide noisy details in smooth areas
        img_gauss = self.apply_gaussian(x, **gaussian_params)
        img_sq_gauss = self.apply_gaussian(x ** 2, **gaussian_params)
        img_std = img_sq_gauss - img_gauss ** 2
        return (x - img_gauss) / tf.sqrt(img_std + div_eps)
