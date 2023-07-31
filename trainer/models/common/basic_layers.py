import tensorflow as tf


class DNNLayer(tf.keras.layers.Layer):
    """A wrapper for basic DNN layers, including Dense, BN, DropOut and l2

    Input shape
      - 2D tensor with shape: ``(batch_size, embedding_size)``.

    Output shape
      - 2D tensor with shape: ``(batch_size, last_dense_layer_size)``.

    """

    def __init__(
        self,
        layer_sizes=(256, 128, 64),
        activation="relu",
        use_bn=True,
        dropout=0.0,
        l2=0.0001,
        **kwargs
    ):
        super(DNNLayer, self).__init__(**kwargs)
        self.model = tf.keras.Sequential()
        for layer_size in layer_sizes:
            self.model.add(
                tf.keras.layers.Dense(
                    layer_size,
                    activation=activation,
                    kernel_regularizer=tf.keras.regularizers.l2(l2),
                )
            )
            if use_bn:
                self.model.add(tf.keras.layers.BatchNormalization())
            if dropout:
                self.model.add(tf.keras.layers.Dropout(dropout))

    def call(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)
