import tensorflow as tf


class FMLayer(tf.keras.layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
    without linear term and bias.

     Input shape
       - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``.

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

    def __init__(self, layer_num=2, l2=0.0001, **kwargs):
        super(CrossNetLayer, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.l2 = l2

    def build(self, input_shape: tf.Tensor, **kwargs):
        dim = input_shape[-1]
        # must provide the name, https://medium.com/dive-into-ml-ai/troubles-with-saving-models-with-custom-layers-in-tensorflow-b31bd5f31a34
        self.ws = [
            self.add_weight(
                name=f"weight_{i}",
                shape=(dim, 1),
                regularizer=tf.keras.regularizers.l2(self.l2),
            )
            for i in range(self.layer_num)
        ]
        self.bs = [
            self.add_weight(name=f"bias_{i}", shape=(dim, 1), initializer="zeros")
            for i in range(self.layer_num)
        ]

    def call(self, inputs, **kwargs):
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


class CrossNetV2Layer(tf.keras.layers.Layer):
    """CrossNet v2 models n-order feature interactions using matrix multiplication

    Input shape
      - 2D tensor with shape: ``(batch_size, embedding_size)``.

    Output shape
      - 2D tensor with shape: ``(batch_size, embedding_size)``.

    References
      - [DCN V2](https://arxiv.org/pdf/2008.13535.pdf)
    """

    def __init__(self, layer_num=2, l2=0.0001, **kwargs):
        super(CrossNetV2Layer, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.l2 = l2

    def build(self, input_shape: tf.Tensor, **kwargs):
        dim = input_shape[-1]
        # must provide the name, https://medium.com/dive-into-ml-ai/troubles-with-saving-models-with-custom-layers-in-tensorflow-b31bd5f31a34
        self.ws = [
            self.add_weight(
                name=f"weight_{i}",
                shape=(dim, dim),
                regularizer=tf.keras.regularizers.l2(self.l2),
            )
            for i in range(self.layer_num)
        ]
        self.bs = [
            self.add_weight(name=f"bias_{i}", shape=(dim, 1), initializer="zeros")
            for i in range(self.layer_num)
        ]

    def call(self, inputs, **kwargs):
        # shape (batch_size, embedding_size, 1)
        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            # shape (batch_size, embedding_size, 1)
            xl_w = tf.matmul(self.ws[i], x_l)
            # shape (batch_size, embedding_size, 1), Hadamard product
            cross_term = x_0 * (xl_w + self.bs[i])
            x_l = cross_term + x_l
        # shape (batch_size, embedding_size)
        # we must specify the axis here. Because there is a None dimension and tf don't know if it should be squeeze or not
        return tf.squeeze(x_l, axis=2)


class CrossNetSimpleMixLayer(tf.keras.layers.Layer):
    """CrossNet v2 models n-order feature interactions using matrix multiplication
    In the simple version, the weight matrix will be decomposed to 2 low-rank matrices

    Input shape
      - 2D tensor with shape: ``(batch_size, embedding_size)``.

    Output shape
      - 2D tensor with shape: ``(batch_size, embedding_size)``.

    References
      - [DCN V2](https://arxiv.org/pdf/2008.13535.pdf)
    """

    def __init__(self, layer_num=2, low_rank=None, l2=0.0001, **kwargs):
        super(CrossNetSimpleMixLayer, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.l2 = l2
        self.low_rank = low_rank

    def build(self, input_shape: tf.Tensor, **kwargs):
        dim = input_shape[-1]
        # by default using dim//4 as the rank number, best setting from the paper
        if self.low_rank is None:
            self.low_rank = dim // 4
        # must provide the name, https://medium.com/dive-into-ml-ai/troubles-with-saving-models-with-custom-layers-in-tensorflow-b31bd5f31a34
        self.Us = [
            self.add_weight(
                name=f"U_{i}",
                shape=(dim, self.low_rank),
                regularizer=tf.keras.regularizers.l2(self.l2),
            )
            for i in range(self.layer_num)
        ]
        self.Vs = [
            self.add_weight(
                name=f"U_{i}",
                shape=(dim, self.low_rank),
                regularizer=tf.keras.regularizers.l2(self.l2),
            )
            for i in range(self.layer_num)
        ]
        self.bs = [
            self.add_weight(name=f"bias_{i}", shape=(dim, 1), initializer="zeros")
            for i in range(self.layer_num)
        ]

    def call(self, inputs, **kwargs):
        # shape (batch_size, embedding_size, 1)
        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            # shape (batch_size, low_rank, 1)
            xl_v = tf.matmul(self.Vs[i], x_l, transpose_a=True)
            # shape (batch_size, embedding_size, 1)
            xl_u = tf.matmul(self.Us[i], xl_v)
            # shape (batch_size, embedding_size, 1), Hadamard product
            cross_term = x_0 * (xl_u + self.bs[i])
            x_l = cross_term + x_l
        # shape (batch_size, embedding_size)
        # we must specify the axis here. Because there is a None dimension and tf don't know if it should be squeeze or not
        return tf.squeeze(x_l, axis=2)


class CrossNetGatingMixLayer(tf.keras.layers.Layer):
    """CrossNet mix models n-order feature interactions using matrix multiplication
    1. Project weight matrix to subspaces and add non-linear transformations
    2. Use MOE to learn feature interactions in different subspaces
    3. Support 3 kinds of gating functions: constant, sigmoid and softmax
    4. Support 2 kinds of activation functions for transformations: identity and tanh

    Input shape
      - 2D tensor with shape: ``(batch_size, embedding_size)``.

    Output shape
      - 2D tensor with shape: ``(batch_size, embedding_size)``.

    References
      - [DCN V2](https://arxiv.org/pdf/2008.13535.pdf)
    """

    def __init__(
        self,
        layer_num=3,
        expert_num=4,
        low_rank=None,
        gate_func="softmax",
        activation="tanh",
        l2=0.0001,
        **kwargs,
    ):
        super(CrossNetGatingMixLayer, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.l2 = l2
        self.expert_num = expert_num
        self.low_rank = low_rank
        self.gate_func = gate_func
        self.activation = activation

    def build(self, input_shape: tf.Tensor, **kwargs):
        dim = input_shape[-1]
        # by default using dim//4 as the rank number, best setting from the paper
        if self.low_rank is None:
            self.low_rank = dim // 4
        # must provide the name, https://medium.com/dive-into-ml-ai/troubles-with-saving-models-with-custom-layers-in-tensorflow-b31bd5f31a34
        self.Us = [
            self.add_weight(
                name=f"U_{i}",
                shape=(self.expert_num, dim, self.low_rank),
                regularizer=tf.keras.regularizers.l2(self.l2),
            )
            for i in range(self.layer_num)
        ]
        self.Cs = [
            self.add_weight(
                name=f"C_{i}",
                shape=(self.expert_num, self.low_rank, self.low_rank),
                regularizer=tf.keras.regularizers.l2(self.l2),
            )
            for i in range(self.layer_num)
        ]
        self.Vs = [
            self.add_weight(
                name=f"V_{i}",
                shape=(self.expert_num, dim, self.low_rank),
                regularizer=tf.keras.regularizers.l2(self.l2),
            )
            for i in range(self.layer_num)
        ]
        self.gates = [
            [tf.keras.layers.Dense(1, use_bias=False) for _ in range(self.expert_num)]
            for _ in range(self.layer_num)
        ]
        self.bs = [
            self.add_weight(name=f"bias_{i}", shape=(dim, 1), initializer="zeros")
            for i in range(self.layer_num)
        ]

    def call(self, inputs, **kwargs):
        # shape (batch_size, embedding_size, 1)
        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            experts = []
            gating_scores = []
            for j in range(self.expert_num):
                # calculate the expert features
                # shape (batch_size, low_rank, 1)
                xl_v = tf.matmul(self.Vs[i][j], x_l, transpose_a=True)
                if self.activation == "tanh":
                    xl_v = tf.nn.tanh(xl_v)
                # shape (batch_size, low_rank, 1)
                xl_c = tf.matmul(self.Cs[i][j], xl_v)
                if self.activation == "tanh":
                    xl_c = tf.nn.tanh(xl_c)
                # shape (batch_size, embedding_size, 1)
                xl_u = tf.matmul(self.Us[i][j], xl_c)
                # shape (batch_size, embedding_size, 1), Hadamard product
                cross_term = x_0 * (xl_u + self.bs[i])
                # shape [(batch_size, embedding_size)]
                experts.append(tf.squeeze(cross_term, axis=2))

                # calculate the gating score
                # shape [(batch_size, 1)]
                gating_scores.append(self.gates[i][j](tf.squeeze(x_l, axis=2)))
            # mix all the experts
            # shape (batch_size, embedding_size, expert_num)
            experts = tf.stack(experts, axis=2)
            # shape (batch_size, expert_num, 1)
            gate_scores = tf.stack(gating_scores, axis=1)
            if self.gate_func == "sigmoid":
                gate_scores = tf.nn.sigmoid(gate_scores)
            elif self.gate_func == "softmax":
                gate_scores = tf.nn.softmax(gate_scores, axis=-1)
            # shape (batch_size, embedding_size, 1)
            moe_output = tf.matmul(experts, gate_scores)
            # shape (batch_size, embedding_size, 1)
            x_l = moe_output + x_l
        # shape (batch_size, embedding_size)
        # we must specify the axis here. Because there is a None dimension and tf don't know if it should be squeeze or not
        return tf.squeeze(x_l, axis=2)


class CINLayer(tf.keras.layers.Layer):
    """Compressed Interaction Network from xDeepFM paper, it contains 3 steps:
    1. outer product to generate field-wise feature interactions
    2. CNN to compress feature vectors
    3. sum pooling to build output

    Input shape
      - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``.

    Output shape
      - 2D tensor with shape: ``(batch_size, feature_map_num)`` ``feature_map_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .

    References
      - [xDeepFM](https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(
        self,
        layer_sizes=(100, 100, 100),
        activation=None,
        l2=0.0001,
        split_half=True,
        **kwargs,
    ):
        super(CINLayer, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        # by default the activation is identity
        self.activation = activation
        self.l2 = l2
        # a faster version, only use half the features as final output and next hidden input
        self.split_half = split_half

    def build(self, input_shape: tf.Tensor):
        self.field_sizes = [input_shape[1]]
        self.emb_dim = input_shape[-1]
        self.filters = []
        for i, size in enumerate(self.layer_sizes):
            self.filters.append(
                # 1D cnn and the filter/kernel size is 1
                tf.keras.layers.Conv1D(
                    size,
                    1,
                    activation=self.activation,
                    kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                )
            )
            if self.split_half:
                if i != len(self.layer_sizes) - 1 and size % 2:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True"
                    )

                self.field_sizes.append(size // 2)
            else:
                self.field_sizes.append(size)

    def call(self, inputs, **kwargs):
        # shape [batch_size, field_size, embedding_size]
        hidden_nn_input = inputs
        final_outputs = []
        for idx, layer_size in enumerate(self.layer_sizes):
            # shape [batch_size, field_size, field_size, embedding_size]
            outer_product = tf.einsum("xiz,xjz->xijz", inputs, hidden_nn_input)
            # shape [batch_size, field_size_0 * field_size_idx, embedding_size]
            merge_product = tf.reshape(
                outer_product,
                shape=[
                    -1,
                    self.field_sizes[0] * self.field_sizes[idx],
                    self.emb_dim,
                ],
            )
            # shape [batch_size, embedding_size, field_size_0 * field_size_idx]
            transposed_product = tf.transpose(merge_product, perm=[0, 2, 1])
            # shape [batch_size, embedding_size, filter_size]
            cur_output = self.filters[idx](transposed_product)
            # shape [batch_size, filter_size, embedding_size]
            cur_output = tf.transpose(cur_output, perm=[0, 2, 1])
            if self.split_half:
                if idx != len(self.layer_sizes) - 1:
                    # shape [batch_size, filter_size // 2, embedding_size]
                    next_hidden, direct_output = tf.split(
                        cur_output, 2 * [layer_size // 2], 1
                    )
                else:
                    direct_output = cur_output
                    next_hidden = None
            else:
                next_hidden = cur_output
                direct_output = cur_output
            final_outputs.append(direct_output)
            hidden_nn_input = next_hidden
        # shape [batch_size, sum(filter_size), embedding_size]
        result = tf.concat(final_outputs, axis=1)
        # shape [batch_size, sum(filter_size)]
        result = tf.reduce_sum(result, -1, keepdims=False)

        return result


class MultiHeadSelfAttentionLayer(tf.keras.layers.Layer):
    """Multi-head self attention layer that models the feature interaction using attention weights

    Input shape
      - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``.

    Output shape
      - 2D tensor with shape: ``(batch_size, field_size, head_dim * head_num)``.

    References
      - [AutoInt](https://arxiv.org/pdf/1810.11921.pdf)
    """

    def __init__(
        self,
        head_dim=32,
        head_num=2,
        use_res=True,
        l2=0,
        **kwargs,
    ):
        super(MultiHeadSelfAttentionLayer, self).__init__(**kwargs)
        self.head_dim = head_dim
        self.head_num = head_num
        self.output_dim = head_dim * head_num
        # use residual or not
        self.use_res = use_res
        self.l2 = l2

    def build(self, input_shape: tf.Tensor):
        embedding_size = input_shape[-1]
        self.W_Query = self.add_weight(
            name="query",
            shape=[embedding_size, self.output_dim],
            regularizer=tf.keras.regularizers.l2(self.l2),
        )
        self.W_Key = self.add_weight(
            name="key",
            shape=[embedding_size, self.output_dim],
            regularizer=tf.keras.regularizers.l2(self.l2),
        )
        self.W_Value = self.add_weight(
            name="value",
            shape=[embedding_size, self.output_dim],
            regularizer=tf.keras.regularizers.l2(self.l2),
        )
        if self.use_res:
            self.W_Res = self.add_weight(
                name="res",
                shape=[embedding_size, self.output_dim],
                regularizer=tf.keras.regularizers.l2(self.l2),
            )

    def call(self, inputs, **kwargs):
        # shape [batch_size, field_size, head_dim * head_num]
        querys = tf.matmul(inputs, self.W_Query)
        keys = tf.matmul(inputs, self.W_Key)
        values = tf.matmul(inputs, self.W_Value)

        # reshape and move the head_num to axis 0
        # shape [head_num, batch_size, field_size, head_dim]
        querys = tf.stack(tf.split(querys, self.head_num, axis=2), axis=0)
        keys = tf.stack(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.stack(tf.split(values, self.head_num, axis=2), axis=0)

        # shape [head_num, batch_size, field_size, field_size]
        weights = tf.matmul(querys, keys, transpose_b=True)
        # scale
        weights /= self.head_dim**0.5
        # shape [head_num, batch_size, field_size, field_size]
        scores = tf.nn.softmax(weights, axis=-1)
        # weighted_sum
        # shape [head_num, batch_size, field_size, head_dim]
        outputs = tf.matmul(scores, values)
        # restore shape
        # shape [[1, batch_size, field_size, head_dim]], head_num lists of tensor
        outputs = tf.split(outputs, self.head_num, axis=0)
        # shape [1, batch_size, field_size, head_dim * head_num]
        outputs = tf.concat(outputs, axis=-1)
        # shape [batch_size, field_size, head_dim * head_num]
        outputs = tf.squeeze(outputs, axis=0)

        if self.use_res:
            # shape [batch_size, field_size, head_dim * head_num]
            outputs += tf.matmul(inputs, self.W_Res)
        outputs = tf.nn.relu(outputs)
        return outputs
