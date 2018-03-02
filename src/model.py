import tensorflow as tf
import features as features_def
from deepnn import DeepNN
from mrifusion.cnn_model import CNN as MriFusionCNN


class Model(DeepNN):
    def __init__(self, is_training, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.is_training = is_training
        self.fuscnn = MriFusionCNN()

    def gen_last_layer(self, ft):
        mri = tf.cast(ft[features_def.MRI], tf.float32)
        mri = tf.reshape(mri, [12] + mri.get_shape()[1:4].as_list() + [1])

        tf.summary.image(
            'input_mri',
            mri[0:1, :, :, 50, :],
        )

        return self.fuscnn.inference(mri)

    def gen_head(self, fc, num_classes, **kwargs):
        return self.fuscnn.cnnutils.inference_conv2(fc, 0.3)
