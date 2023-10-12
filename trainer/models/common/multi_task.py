import tensorflow as tf


class MMOELayer(tf.keras.layers.Layer):
    """A wrapper for basic dense layers, including Dense, BN, DropOut and l2

    Input shape
      - 2D tensor with shape: ``(batch_size, embedding_size)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, gate_num, last_dense_layer_size)``.

    """

    def __init__(
        self,
        expert_num=2,
        gate_num=2,
        layer_sizes=(256, 128, 64),
        activation="relu",
        use_bn=True,
        dropout=0.0,
        l2=0.0001,
        **kwargs
    ):
        super(MMOELayer, self).__init__(**kwargs)
        self.experts = []
        for _ in range(expert_num):
            expert = tf.keras.Sequential(name="expert")
            for layer_size in layer_sizes:
                expert.add(
                    tf.keras.layers.Dense(
                        layer_size,
                        activation=activation,
                        kernel_regularizer=tf.keras.regularizers.l2(l2),
                    )
                )
            if use_bn:
                expert.add(tf.keras.layers.BatchNormalization())
            if dropout:
                expert.add(tf.keras.layers.Dropout(dropout))
            self.experts.append(expert)
        self.gates = [
            tf.keras.layers.Dense(expert_num, activation="softmax")
            for _ in range(gate_num)
        ]

    def call(self, inputs, **kwargs):
        # shape [batch_size, expert_num, last_dense_layer_size]
        experts_output = tf.stack(
            [expert(inputs, **kwargs) for expert in self.experts], axis=1
        )
        # shape [batch_size, gate_num, expert_num]
        gate_output = tf.stack([gate(inputs, **kwargs) for gate in self.gates], axis=1)
        # shape [batch_size, gate_num, last_dense_layer_size]
        return tf.matmul(gate_output, experts_output)
