"""
Tensorflow layer compositions
"""
import tensorflow as tf


def conv_layer(x, layer, name, reuse=None, weight_decay=None):
    """
    Creates and returns a new convolutional layer.
    
    For a 2D convolution specify:
    
    .. code-block:: python
    
        - type: 'conv'
          kernel: [3, 3]
          feature_maps: 32
          stride: [1, 1]
          activation: 'tf.nn.relu'
          padding: 'same'
          batch_norm: True
          
    For a 3D convolution, e.g. specify:
          
    .. code-block:: python
    
        - type: 'conv'
          kernel: [3, 3, 3]
          feature_maps: 32
          stride: [1, 1, 1]
          activation: 'tf.nn.relu'
          padding: 'same'
          batch_norm: True
    
    The important thing is to make sure that all the list hold 2 values for 2D conv, 
    and 3 for a 3D conv.
          
    Args:
        x (Tensor): Input 
        layer (dict): Layer configuration as outlined above.
        name (string): Layer name
        reuse (bool): Reuse layers
        weight_decay (object): The tf.layers Weight decay function 

    Returns:
        Returns a new convolutional layer.
    """
    with tf.name_scope('conv'):
        if len(layer['kernel']) == 2:
            fn = tf.layers.conv2d
        elif len(layer['kernel']) == 3:
            fn = tf.layers.conv3d
        else:
            raise RuntimeError(str(len(layer['kernel'])) + "d convolution not supported`")

        return fn(
            inputs=x, filters=layer['feature_maps'], kernel_size=layer['kernel'],
            strides=layer['stride'], padding=layer['padding'], activation=eval(layer['activation']),
            kernel_regularizer=weight_decay, reuse=reuse, name=name)


def deconv_layer(x, layer, name, reuse=None, weight_decay=None):
    """
    Creates and returns a new deconvolutional layer.

    For a 2D deconvolution specify:

    .. code-block:: python

        - type: 'deconv'
          kernel: [3, 3]
          feature_maps: 32
          stride: [1, 1]
          activation: 'tf.nn.relu'
          padding: 'same'
          batch_norm: True

    For a 3D convolution, e.g. specify:

    .. code-block:: python

        - type: 'deconv'
          kernel: [3, 3, 3]
          feature_maps: 32
          stride: [1, 1, 1]
          activation: 'tf.nn.relu'
          padding: 'same'
          batch_norm: True

    The important thing is to make sure that all the list hold 2 values for 2D conv, 
    and 3 for a 3D conv.

    Args:
        x (Tensor): Input 
        layer (dict): Layer configuration as outlined above.
        name (string): Layer name
        reuse (bool): Reuse layers
        weight_decay (object): The tf.layers Weight decay function 

    Returns:
        Returns a new deconvolutional layer.
    """

    with tf.name_scope('deconv'):
        if len(layer['kernel']) == 2:
            fn = tf.layers.conv2d_transpose
        elif len(layer['kernel']) == 3:
            fn = tf.layers.conv3d_transpose
        else:
            raise RuntimeError(str(len(layer['kernel'])) + "d convolution not supported`")

        return fn(inputs=x, filters=layer['feature_maps'], kernel_size=layer['kernel'],
                  strides=layer['stride'], padding=layer['padding'],
                  activation=eval(layer['activation']),
                  kernel_regularizer=weight_decay, reuse=reuse, name=name)


def max_pooling_layer(x, layer):
    """
    Creates and returns a new max pooling layer.

    For "2D" max pooling specify:

    .. code-block:: python

        - type: 'max_pool'
          window: [2, 2]
          stride: [1, 1]

    For "3D" max pooling, e.g. specify:

    .. code-block:: python

        - type: 'max_pool'
          window: [2, 2, 2]
          stride: [1, 1, 1]

    The important thing is to make sure that all the list hold 2 values for 2D conv, 
    and 3 for a 3D conv.

    Args:
        x (Tensor): Input
        layer (dict): Layer configuration as outlined above.

    Returns:
        Returns a new max pooling layer.
    """
    with tf.name_scope('max_pool'):
        if len(layer['window']) == 2:
            fn = tf.layers.max_pooling2d
        elif len(layer['window']) == 3:
            fn = tf.layers.max_pooling3d
        else:
            raise RuntimeError(str(len(layer['window'])) + "d max pooling not supported`")

        return fn(inputs=x, pool_size=layer['window'], strides=layer['stride'],
                  padding="same")


def dense_layer(x, layer, name, reuse=None, weight_decay=None):
    """
    Creates and returns a new dense layer. The input gets flattened using :code:`
    tf.contrib.layers.flatten`, which preserves the batch dimension.

    Configuration dict:

    .. code-block:: python

      - type: 'dense'
        units: 200
        activation: 'tf.nn.relu'
        batch_norm: True

    Args:
        x (Tensor): Input 
        layer (dict): Layer configuration as outlined above.
        name (string): Layer name
        reuse (bool): Reuse layers
        weight_decay (object): The tf.layers Weight decay function 

    Returns:
        Returns a new dense layer.
    """
    with tf.name_scope('dense'):
        return tf.layers.dense(inputs=tf.contrib.layers.flatten(x), units=layer['units'],
                               activation=eval(layer['activation']),
                               kernel_regularizer=weight_decay, reuse=reuse, name=name)


def build_architecture(x, architecture, training, scope, reuse=None, weight_decay=None):
    """
    Creates a hierarchy of layers according to a provided list odf dict objects
    (:code:`architecture`).
    
    For example, a possible deep encoding for a convolutional autoencoder might be:
    
    .. code-block:: python
        
        encoding:
          - type: 'conv'
            kernel: [3, 3]
            feature_maps: 32
            stride: [1, 1]
            activation: 'tf.nn.relu'
            padding: 'same'
            batch_norm: True
    
          - type: 'max_pool'
            window: [2, 2]
            stride: [1, 1]

          - type: 'dropout'
            drop: 0.1
    
          - type: 'conv'
            kernel: [5, 5]
            feature_maps: 64
            stride: [1, 1]
            activation: 'tf.nn.relu'
            padding: 'same'
            batch_norm: True
    
          - type: 'max_pool'
            window: [2, 2]
            stride: [1, 1]
    
          - type: 'conv'
            kernel: [7, 7]
            feature_maps: 128
            stride: [1, 1]
            activation: 'tf.nn.relu'
            padding: 'same'
            batch_norm: True
    
          - type: 'max_pool'
            window: [2, 2]
            stride: [1, 1]
            
    or
    
    .. code-block:: python
    
        decoding:
         - type: 'dense'
           units: 81
           activation: 'tf.nn.relu'
           batch_norm: False

         - type: 'reshape'
           shape: [-1, 9, 9, 1]
           
         - type: 'deconv'
           kernel: [3, 3]
           feature_maps: 32
           stride: [1, 1]
           activation: 'tf.nn.relu'
           padding: 'same'
           batch_norm: True
            
    Args:
        x (Tensor): Input 
        architecture: List of dicts specifying the architecture as outlined above. 
        training: Used to control dropout. Set it to :code:`True` during training and 
            to :code:`False` for everything other
        scope (string): Scope name 
        reuse (bool): Reuse layers
        weight_decay (object): The tf.layers Weight decay function 

    Returns:

    """
    with tf.variable_scope(scope):
        for i, layer in enumerate(architecture):
            if 'batch_norm' in layer and layer['batch_norm']:
                layer['activation_bn'] = layer['activation']
                layer['activation'] = "tf.identity"

            if layer['type'] == 'reshape':
                x = tf.reshape(x, layer['shape'], name=str(i))
            elif layer['type'] == "conv":
                x = conv_layer(
                    x=x, layer=layer, weight_decay=weight_decay, reuse=reuse, name=str(i))
            elif layer['type'] == "deconv":
                x = deconv_layer(
                    x=x, layer=layer, weight_decay=weight_decay, reuse=reuse, name=str(i))
            elif layer['type'] == "max_pool":
                x = max_pooling_layer(x=x, layer=layer)
            elif layer['type'] == "dense":
                x = dense_layer(
                    x=x, layer=layer, weight_decay=weight_decay, reuse=reuse, name=str(i))
            elif layer['type'] == "dropout":
                x = tf.layers.dropout(x, rate=layer['drop'], training=training)
            elif layer["type"] == "softmax":
                x = tf.layers.softmax(logits=x, scope="softmax")
            else:
                raise RuntimeError("Layer typ `" + layer['type'] + "` not supported")

            # Apply batch norm
            if 'batch_norm' in layer and layer['batch_norm']:
                with tf.variable_scope("batch_norm"):
                    x = tf.layers.batch_normalization(
                        x, training=training, reuse=reuse, name=str(i))
                    x = eval(layer['activation_bn'])(x)

        return x
