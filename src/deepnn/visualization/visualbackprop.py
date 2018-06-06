import numpy as np
import tensorflow as tf


def upsample_to(tensor, output_shape):
    f = np.ones([2, 2, 2, 1, 1], dtype=np.float32)
    return tf.nn.conv3d_transpose(
        tensor,
        f,
        output_shape=output_shape,
        strides=[1, 2, 2, 2, 1],
    )


def average_ft_map(tensor):
    tensor = tf.reduce_mean(tensor, axis=4, keepdims=True)
    t_min = tf.reduce_min(tensor, axis=[1, 2, 3], keepdims=True)
    t_max = tf.reduce_max(tensor, axis=[1, 2, 3], keepdims=True)
    return (tensor - t_min) / (t_max - t_min + 1e-6)


class VisualBackProp:
    def get_visualization(self, all_convs_outputs):
        start_conv_index = 2
        visualbackprop_output = average_ft_map(
            all_convs_outputs[-start_conv_index])
        for next_ft_map in list(
            reversed(all_convs_outputs)
        )[start_conv_index:]:
            next_ft_map = average_ft_map(next_ft_map)
            visualbackprop_output = upsample_to(
                visualbackprop_output,
                tf.shape(next_ft_map),
            )
            visualbackprop_output *= next_ft_map
        return visualbackprop_output
