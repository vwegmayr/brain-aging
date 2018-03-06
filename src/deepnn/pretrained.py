import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework.errors_impl import NotFoundError
from modules.models.utils import custom_print


class PretrainedNetwork(object):
    """
    Currently only supports inception v3
    """
    def __init__(
        self,
        model_path='data/models/classify_image_graph_def.pb',
    ):
        self.model_info = PretrainedNetwork.inception_v3_model_info()
        self.model_path = model_path

    @staticmethod
    def inception_v3_model_info():
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        # bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_name = 'pool_3:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128
        return {
            'data_url': data_url,
            'bottleneck_tensor_name': bottleneck_tensor_name,
            'bottleneck_tensor_size': bottleneck_tensor_size,
            'input_width': input_width,
            'input_height': input_height,
            'input_depth': input_depth,
            'resized_input_tensor_name': resized_input_tensor_name,
            'model_file_name': model_file_name,
            'input_mean': input_mean,
            'input_std': input_std,
            'quantize_layer': False,
        }

    def resize_input_images(self, images):
        assert(len(images.get_shape().as_list()) == 4)
        assert(images.get_shape().as_list()[-1] == 1)
        # Set images as array of uint8
        images = tf.cast(10.0 * images + 128, tf.uint8)
        # Set number of channels
        images_channels = []
        for i in range(self.model_info['input_depth']):
            images_channels.append(images)
        images = tf.concat(images_channels, 3)
        # Resize
        resize_shape = tf.stack([
            self.model_info['input_height'],
            self.model_info['input_width'],
        ])
        resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
        resized_image = tf.image.resize_bilinear(images,
                                                 resize_shape_as_int)
        offset_image = tf.subtract(resized_image,
                                   self.model_info['input_mean'])
        return tf.multiply(offset_image, 1.0 / self.model_info['input_std'])

    def create_graph(self, resized_image_tensor):
        input_tensor_name = self.model_info['resized_input_tensor_name']
        try:
            with gfile.FastGFile(self.model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                custom_print('[Pretrained] Model loaded: ', self.model_path)
                model_features = tf.import_graph_def(
                    graph_def,
                    name='inceptionv3',
                    input_map={
                        input_tensor_name: resized_image_tensor,
                    },
                    return_elements=[
                        self.model_info['bottleneck_tensor_name'],
                    ],
                )
            return tf.reshape(
                model_features[0],
                [-1, self.model_info['bottleneck_tensor_size']],
            )
        except NotFoundError:
            custom_print((
                'Unable to open file %s; please download the model' +
                ' there (%s)') % (
                    self.model_path, self.model_info['data_url'])
            )
            exit(0)
