import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    """Position embedding is composed of a positional encoding layer and a normal embedding layer

    Input shape
      - 2D tensor with shape: ``(batch_size, sequence_length)``.

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
        length = tf.shape(inputs)[1]
        embedded_tokens = self.token_emb(inputs)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        embedded_tokens *= tf.math.sqrt(tf.cast(self.dim, tf.float32))
        return embedded_tokens + self.pos_enc[tf.newaxis, :length, :]

    # Pass mask from token_emb, https://www.tensorflow.org/guide/keras/understanding_masking_and_padding#supporting_masking_in_your_custom_layers
    def compute_mask(self, *args, **kwargs):
        return self.token_emb.compute_mask(*args, **kwargs)

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
        dropout=0.0,
        use_bias=True,
        **kwargs,
    ):
        super(MultiHeadSelfAttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.head_num = head_num
        self.key_output_dim = key_dim * head_num
        self.val_output_dim = val_dim * head_num if val_dim else self.key_output_dim
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.use_bias = use_bias

    def build(self, input_shape: tf.Tensor):
        embedding_size = input_shape[-1]
        self.W_Query = self.add_weight(
            name="weight_query",
            shape=[embedding_size, self.key_output_dim],
        )
        self.W_Key = self.add_weight(
            name="weight_key",
            shape=[embedding_size, self.key_output_dim],
        )
        self.W_Value = self.add_weight(
            name="weight_value",
            shape=[embedding_size, self.val_output_dim],
        )
        self.W_Output = self.add_weight(
            name="weight_output",
            shape=[self.val_output_dim, embedding_size],
        )
        if self.use_bias:
            self.B_Query = self.add_weight(
                "bias_query",
                shape=[
                    self.key_output_dim,
                ],
                initializer="zeros",
            )
            self.B_Key = self.add_weight(
                "bias_key",
                shape=[
                    self.key_output_dim,
                ],
                initializer="zeros",
            )
            self.B_Value = self.add_weight(
                "bias_value",
                shape=[
                    self.val_output_dim,
                ],
                initializer="zeros",
            )
            self.B_Output = self.add_weight(
                "bias_output",
                shape=[
                    embedding_size,
                ],
                initializer="zeros",
            )

    def call(
        self,
        query,
        key,
        value,
        training=False,
        use_causal_mask=False,
    ):
        # shape [batch_size, query_length, key_length]
        mask = self._compute_attention_mask(
            query, key, value, use_causal_mask=use_causal_mask
        )
        # shape [batch_size, query_length, key_dim * head_num]
        querys = tf.matmul(query, self.W_Query)
        # shape [batch_size, key_length, key_dim * head_num]
        keys = tf.matmul(key, self.W_Key)
        # shape [batch_size, key_length, val_dim * head_num]
        values = tf.matmul(value, self.W_Value)

        if self.use_bias:
            querys = tf.nn.bias_add(querys, self.B_Query)
            keys = tf.nn.bias_add(keys, self.B_Key)
            values = tf.nn.bias_add(values, self.B_Value)

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
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.B_Output)
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

    def get_config(self):
        # to make save and load a model using custom layer possible
        config = super().get_config()
        config.update(
            {
                "head_num": self.head_num,
                "key_dim": self.key_dim,
                "val_dim": self.val_dim,
                "dropout": self.dropout,
                "use_bias": self.use_bias,
            }
        )
        return config


class FeedForward(tf.keras.layers.Layer):
    """feed forward layer is composed of
        relu dense layer -> linear dense layer -> add & layer normalization layer

    Input shape
      - 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    References
        - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(self, ff_dim=2048, dropout=0.1, model_dim=512, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        # use the Add layer to ensure that Keras masks are propagated (the + operator does not).
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()
        # dropout layers are applied before residual and normalization layer
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dense1 = tf.keras.layers.Dense(ff_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(model_dim)

    def call(self, input, training=False):
        dense_output = self.dropout1(self.dense2(self.dense1(input)), training=training)
        output = self.norm(self.add([input, dense_output]), training=training)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
            }
        )
        return config


class Encoder(tf.keras.layers.Layer):
    """The Transformer Encoder consists of
    multi-head attention layer -> add & normalization layer ->
        MLP layer -> add & normalization layer

    Input shape
      - 3D tensor with shape ``(batch_size, token_length, embedding_size)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, token_length, embedding_size)``.

    References
        - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(
        self,
        model_dim=512,
        ff_dim=2048,
        dropout=0.1,
        head_num=8,
        **kwargs,
    ):
        super(Encoder, self).__init__(**kwargs)
        assert model_dim % head_num == 0, "model_dim need to be divisible by head_num"
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.head_num = head_num
        # use the Add layer to ensure that Keras masks are propagated (the + operator does not).
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()
        self.ff = FeedForward(ff_dim=ff_dim, dropout=dropout, model_dim=model_dim)
        # get key_dim
        key_dim = model_dim // head_num
        self.attention = MultiHeadSelfAttentionLayer(
            head_num=head_num, key_dim=key_dim, dropout=dropout
        )

    def call(self, input, training=False):
        # shape [batch_size, token_length, embedding_size]
        attention_output = self.norm(
            self.add(
                [input, self.attention(input, input, input, training=training)],
            ),
            training=training,
        )

        return self.ff(attention_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_num": self.head_num,
                "model_dim": self.model_dim,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
                "head_num": self.head_num,
            }
        )
        return config


class Decoder(tf.keras.layers.Layer):
    """The Transformer Decoder consists of
    self attention layer -> add & normalization layer ->
        cross attention layer -> add & normalization layer ->
            MLP layer -> add & normalization layer

    Input shape
      Notice that the key and value are from the output of encoder
      - encoder_output: 3D tensor with shape ``(batch_size, key_length, embedding_size)``.
      - decoder_input: 3D tensor with shape ``(batch_size, query_length, embedding_size)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, token_length, embedding_size)``.

    References
        - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(
        self,
        model_dim=512,
        ff_dim=2048,
        dropout=0.1,
        head_num=8,
        **kwargs,
    ):
        super(Decoder, self).__init__(**kwargs)
        assert model_dim % head_num == 0, "model_dim need to be divisible by head_num"
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.head_num = head_num
        # use the Add layer to ensure that Keras masks are propagated (the + operator does not).
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.ff = FeedForward(ff_dim=ff_dim, dropout=dropout, model_dim=model_dim)
        # get key_dim
        key_dim = model_dim // head_num
        self.self_attention = MultiHeadSelfAttentionLayer(
            head_num=head_num, key_dim=key_dim, dropout=dropout
        )
        self.cross_attention = MultiHeadSelfAttentionLayer(
            head_num=head_num, key_dim=key_dim, dropout=dropout
        )

    def call(self, encoder_output, decoder_input, training=False):
        # shape [batch_size, query_length, embedding_size]
        # use causal_mask here
        self_attention_output = self.norm1(
            self.add1(
                [
                    decoder_input,
                    self.self_attention(
                        decoder_input,
                        decoder_input,
                        decoder_input,
                        training=training,
                        use_causal_mask=True,
                    ),
                ]
            ),
            training=training,
        )
        # query is from the output of previous self attention layer
        cross_attention_output = self.norm2(
            self.add2(
                [
                    self_attention_output,
                    self.cross_attention(
                        self_attention_output,
                        encoder_output,
                        encoder_output,
                        training=training,
                    ),
                ]
            ),
            training=training,
        )
        return self.ff(cross_attention_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_num": self.head_num,
                "model_dim": self.model_dim,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
                "head_num": self.head_num,
            }
        )
        return config


class Transformer(tf.keras.layers.Layer):
    """The Transformer consists of multiple layers of encoder and decoder

    Input shape
      - 3D tensor with shape ``(batch_size, token_length)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, token_length, vocab_size_logits)``.

    References
        - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(
        self,
        src_vocab_size,
        target_vocab_size,
        seq_length,
        layer_num=6,
        model_dim=512,
        ff_dim=2048,
        dropout=0.1,
        head_num=8,
        **kwargs,
    ):
        super(Transformer, self).__init__(**kwargs)
        assert model_dim % head_num == 0, "model_dim need to be divisible by head_num"
        self.src_vocab_size = src_vocab_size
        self.target_vocab_size = target_vocab_size
        self.seq_length = seq_length
        self.layer_num = layer_num
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.head_num = head_num

        self.src_emb = PositionalEmbedding(
            vocab_size=src_vocab_size, length=seq_length, dim=model_dim
        )
        self.target_emb = PositionalEmbedding(
            vocab_size=target_vocab_size, length=seq_length, dim=model_dim
        )
        self.encoders = [
            Encoder(
                model_dim=model_dim, ff_dim=ff_dim, dropout=dropout, head_num=head_num
            )
            for _ in range(layer_num)
        ]
        self.encoder_dropout = tf.keras.layers.Dropout(dropout)
        self.decoders = [
            Decoder(
                model_dim=model_dim, ff_dim=ff_dim, dropout=dropout, head_num=head_num
            )
            for _ in range(layer_num)
        ]
        self.decoder_dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(
            target_vocab_size, activation=tf.keras.activations.softmax
        )

    def call(self, src, target, training=False):
        # shape [batch_size, token_length]
        src_emb = self.encoder_dropout(self.src_emb(src, training=training))
        target_emb = self.decoder_dropout(self.target_emb(target, training=training))
        # shape [batch_size, token_length, model_dim]
        for encoder in self.encoders:
            src_emb = encoder(src_emb, training=training)
        for decoder in self.decoders:
            target_emb = decoder(src_emb, target_emb, training=training)
        # shape [batch_size, token_length, vocab_size_logits]
        return self.dense(target_emb)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "src_vocab_size": self.src_vocab_size,
                "target_vocab_size": self.target_vocab_size,
                "layer_num": self.layer_num,
                "seq_length": self.seq_length,
                "model_dim": self.model_dim,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
                "head_num": self.head_num,
            }
        )
        return config


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, model_dim, warmup_steps=4000):
        super().__init__()

        self.model_dim = model_dim
        self.model_dim = tf.cast(self.model_dim, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "warmup_steps": self.warmup_steps,
            }
        )
        return config


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction="none"
    )
    loss = loss_object(label, pred)

    # mask indices where label == 0 (padding)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    # get the prediction index for target token
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)
