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
