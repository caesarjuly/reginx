import tensorflow as tf


class FMLayer(tf.keras.layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
    without linear term and bias.

     Input shape
       - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

     Output shape
       - 2D tensor with shape: ``(batch_size, 1)``.

     References
       - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(inputs * inputs, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)

        return cross_term


class CrossNetLayer(tf.keras.layers.Layer):
    """CrossNet models n-order feature interactions using matrix multiplication

    Input shape
      - 2D tensor with shape: ``(batch_size, embedding_size)``.

    Output shape
      - 2D tensor with shape: ``(batch_size, embedding_size)``.

    References
      - [DCN](https://arxiv.org/pdf/1708.05123.pdf)
    """

    def __init__(self, layer_num=2, **kwargs):
        super(CrossNetLayer, self).__init__(**kwargs)
        self.layer_num = layer_num

    def build(self, input_shape: tf.Tensor):
        dim = input_shape[-1]
        # must provide the name, https://medium.com/dive-into-ml-ai/troubles-with-saving-models-with-custom-layers-in-tensorflow-b31bd5f31a34
        self.ws = [
            self.add_weight(name=f"weight_{i}", shape=(dim, 1))
            for i in range(self.layer_num)
        ]
        self.bs = [
            self.add_weight(name=f"bias_{i}", shape=(dim, 1), initializer="zeros")
            for i in range(self.layer_num)
        ]

    def call(self, inputs):
        """we can also do the matrix multiplication on x_0 and x_l first, but this will explode the memory/computation consumption
        An example:
          # shape (batch_size, embedding_size, embedding_size)
          x_cross = tf.matmul(a, b, transpose_b=True)
          # shape (batch_size, embedding_size, 1)
          x_l = tf.matmul(x_cross, self.weights[i])
        """
        # shape (batch_size, embedding_size, 1)
        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            # shape (batch_size, 1, 1)
            xl_w = tf.matmul(x_l, self.ws[i], transpose_a=True)
            # shape (batch_size, embedding_size, 1)
            cross_term = tf.matmul(x_0, xl_w)
            x_l = cross_term + self.bs[i] + x_l
        # shape (batch_size, embedding_size)
        # we must specify the axis here. Because there is a None dimension and tf don't know if it should be squeeze or not
        return tf.squeeze(x_l, axis=2)
