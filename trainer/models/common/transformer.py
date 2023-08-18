import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    """Position embedding is composed of a positional encoding layer and a normal embedding layer

    Input shape
      - two 2D tensor with shape: ``(batch_size, sequence_length)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    References
        - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(self, vocab_size, length=100, dim=512, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        assert length % 2 == 0, "Output dimension needs to be an even integer"
        self.length = length
        self.dim = dim
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=dim, mask_zero=True
        )
        # positional embedding layer: a matrix of hard-coded sine values
        self.pos_enc = self.positional_encoding()

    def positional_encoding(self, n=10000):
        """Position encoding using sine and cosine functions to represent the order information

        Args:
            n: Constant for the sinusoidal functions

        Output shape
        - 2D tensor with shape: ``(length, dim)``.
        """
        half_dim = self.dim // 2
        # shape [length, 1] column vector
        pos = tf.reshape(tf.range(self.length, dtype=tf.float32), [-1, 1])
        # shape [1, half_dim] row vector
        i = tf.reshape(tf.range(half_dim, dtype=tf.float32), [1, -1])
        # n**(-2*i/dim)
        denom = tf.math.pow(n, -i / half_dim)
        # shape [length, half_dim]
        args = pos * denom
        # shape [length, half_dim, 1] generate even dimensions
        sin = tf.expand_dims(tf.math.sin(args), axis=-1)
        # shape [length, half_dim, 1] generate odd dimensions
        cos = tf.expand_dims(tf.math.cos(args), axis=-1)
        # shape [length, half_dim, 2] -> [length, dim], concat and interleave
        pe = tf.reshape(tf.concat([sin, cos], axis=-1), [self.length, self.dim])
        return pe

    def call(self, inputs, **kwargs):
        embedded_tokens = self.token_emb(inputs)
        return embedded_tokens + self.pos_enc

    # Pass mask from token_emb, https://www.tensorflow.org/guide/keras/understanding_masking_and_padding#supporting_masking_in_your_custom_layers
    def compute_mask(self, inputs, mask=None):
        return self.token_emb.compute_mask(inputs, mask=mask)

    def get_config(self):
        # to make save and load a model using custom layer possible
        config = super().get_config()
        config.update(
            {
                "length": self.length,
                "vocab_size": self.vocab_size,
                "dim": self.dim,
            }
        )
        return config


class MultiHeadSelfAttentionLayer(tf.keras.layers.Layer):
    """Multi-head self attention layer that models the token interactions

    Input shape
      - query: 3D tensor with shape ``(batch_size, query_length, embedding_size)``.
      - key: 3D tensor with shape ``(batch_size, key_length, embedding_size)``.
      - value: 3D tensor with shape ``(batch_size, key_length, embedding_size)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, query_length, embedding_size)``.

    References
        - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(
        self,
        head_num=8,
        key_dim=64,
        val_dim=None,
        dropout=0.1,
        **kwargs,
    ):
        super(MultiHeadSelfAttentionLayer, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.head_num = head_num
        self.key_output_dim = key_dim * head_num
        self.val_output_dim = val_dim * head_num if val_dim else self.key_output_dim
        self.dropout = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape: tf.Tensor):
        embedding_size = input_shape[-1]
        self.W_Query = self.add_weight(
            name="query",
            shape=[embedding_size, self.key_output_dim],
        )
        self.W_Key = self.add_weight(
            name="key",
            shape=[embedding_size, self.key_output_dim],
        )
        self.W_Value = self.add_weight(
            name="value",
            shape=[embedding_size, self.val_output_dim],
        )
        self.W_Output = self.add_weight(
            name="output",
            shape=[self.val_output_dim, embedding_size],
        )

    def call(
        self,
        query,
        key,
        value,
        use_causal_mask=False,
        training=None,
    ):
        # shape [head_num, batch_size, query_length, key_length]
        mask = self._compute_attention_mask(
            query, key, value, use_causal_mask=use_causal_mask
        )
        # shape [batch_size, query_length, key_dim * head_num]
        querys = tf.matmul(query, self.W_Query)
        # shape [batch_size, key_length, key_dim * head_num]
        keys = tf.matmul(key, self.W_Key)
        # shape [batch_size, key_length, val_dim * head_num]
        values = tf.matmul(value, self.W_Value)

        # reshape and move the head_num to axis 0
        # shape [head_num, batch_size, query_length, key_dim]
        querys = tf.stack(tf.split(querys, self.head_num, axis=2), axis=0)
        # shape [head_num, batch_size, key_length, key_dim]
        keys = tf.stack(tf.split(keys, self.head_num, axis=2), axis=0)
        # shape [head_num, batch_size, key_length, val_dim]
        values = tf.stack(tf.split(values, self.head_num, axis=2), axis=0)

        # shape [head_num, batch_size, query_length, key_length]
        weights = tf.matmul(querys, keys, transpose_b=True)
        # scale
        weights /= self.key_dim**0.5
        # mask
        if mask is not None:
            adder = (1.0 - tf.cast(mask, weights.dtype)) * -1e9
            weights += adder
        scores = tf.nn.softmax(weights, axis=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        scores = self.dropout(scores, training=training)

        # weighted_sum
        # shape [head_num, batch_size, query_length, val_dim]
        outputs = tf.matmul(scores, values)
        # restore shape
        # shape [[1, batch_size, query_length, val_dim]], head_num lists of tensor
        outputs = tf.split(outputs, self.head_num, axis=0)
        # shape [1, batch_size, query_length, val_dim * head_num]
        outputs = tf.concat(outputs, axis=-1)
        # shape [batch_size, query_length, val_dim * head_num]
        outputs = tf.squeeze(outputs, axis=0)

        # shape [batch_size, query_length, embedding_size]
        outputs = tf.matmul(outputs, self.W_Output)
        return outputs

    def _compute_attention_mask(self, query, key, value, use_causal_mask=False):
        """Computes the attention mask, using the Keras masks of the inputs.

            * The `query`'s mask is reshaped from [batch_size, query_length] to [batch_size, query_length, 1].
            * The `value`'s mask is reshaped from [batch_size, key_length] to [batch_size, 1, key_length].
            * The `key`'s mask is reshaped from [batch_size, key_length] to [batch_size, 1, key_length]. The `key`'s
              mask is ignored if `key` is `None` or if `key is value`.
            * If `use_causal_mask=True`, then the causal mask is computed. Its shape
              is [1, query_length, key_length].

            All defined masks are merged using a logical AND operation (`&`).

        Input shape
          - query: 3D tensor with shape ``(batch_size, query_length, embedding_size)``.
          - key: 3D tensor with shape ``(batch_size, key_length, embedding_size)``.
          - value: 3D tensor with shape ``(batch_size, key_length, embedding_size)``.

        Output shape
          - 3D tensor with shape: ``(batch_size, query_length, embedding_size)``.
        """
        query_mask = getattr(query, "_keras_mask", None)
        value_mask = getattr(value, "_keras_mask", None)
        key_mask = getattr(key, "_keras_mask", None)
        auto_mask = None
        if query_mask is not None:
            query_mask = tf.cast(query_mask, tf.bool)  # defensive casting
            # B = batch size, T = max query length
            auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]
        if value_mask is not None:
            value_mask = tf.cast(value_mask, tf.bool)  # defensive casting
            # B = batch size, S == max value length
            mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if key_mask is not None:
            key_mask = tf.cast(key_mask, tf.bool)  # defensive casting
            # B == batch size, S == max key length == max value length
            mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if use_causal_mask:
            # the shape of the causal mask is [1, T, S]
            mask = self._compute_causal_mask(query, value)
            auto_mask = mask if auto_mask is None else auto_mask & mask
        return auto_mask

    def _compute_causal_mask(self, query, value=None):
        """Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean `Tensor` equal to:

        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]
        ```

        Args:
            query: query `Tensor` of shape `(batch_size, query_length, ...)`.
            value: value `Tensor` of shape `(batch_size, key_length, ...)` (optional, defaults to
                query).

        Returns:
            mask: a boolean `Tensor` of shape [1, T, S] containing a lower
                triangular matrix of shape [T, S].
        """
        q_seq_length = tf.shape(query)[1]
        v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
        return tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0
        )
