import tensorflow as tf
from base import NetworkHeadBase
import src.features as ft_def


class AutoencoderHead(NetworkHeadBase):
    def __init__(
        self,
        # NetworkHeadBase arguments
        name,
        model,
        last_layer,
        features,
        # Custom arguments (from config file)
        # ...
        # Args passed to parent
        **kwargs
    ):
        self.labels = features[ft_def.MRI]
        self.predictions = tf.reshape(
            model.gen_deconv_head(last_layer),
            [-1] + self.labels.get_shape().as_list()[1:],
        )
        self.loss = tf.losses.mean_squared_error(
            self.labels,
            self.predictions,
        )
        super(AutoencoderHead, self).__init__(
            name=name,
            model=model,
            last_layer=last_layer,
            features=features,
            **kwargs
        )

    def get_logged_training_variables(self):
        training_variables = \
            super(AutoencoderHead, self).get_logged_training_variables()
        return training_variables
