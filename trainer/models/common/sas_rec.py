import tensorflow as tf

from trainer.models.common.transformer import FeedForward, MultiHeadSelfAttentionLayer


class PositionalEmbedding(tf.keras.layers.Layer):
    """SASRec embedding is composed of a positional embedding layer and a normal embedding layer

    Input shape
      - token index 2D tensor with shape: ``(batch_size, sequence_length)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    References
        - [Self-Attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf)
    """

    def __init__(self, token_embedding, seq_length=50, dim=50, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.length = seq_length
        self.dim = dim
        self.token_emb = token_embedding
        self.position_emb = tf.keras.layers.Embedding(
            input_dim=seq_length, output_dim=dim
        )

    def call(self, inputs, **kwargs):
        length = tf.shape(inputs)[1]
        embedded_tokens = self.token_emb(inputs)
        embedded_positions = self.position_emb(tf.range(length))
        # This factor sets the relative scale of the embedding and positonal_encoding.
        embedded_tokens *= tf.math.sqrt(tf.cast(self.dim, tf.float32))
        return embedded_tokens + embedded_positions[tf.newaxis, :, :]

    # Pass mask from token_emb, https://www.tensorflow.org/guide/keras/understanding_masking_and_padding#supporting_masking_in_your_custom_layers
    def compute_mask(self, inputs, mask=None):
        return self.token_emb.compute_mask(inputs, mask=mask)


class SASRecBlock(tf.keras.layers.Layer):
    """SASRec block is a stack of self attention layer + MLP + layer norm + residual layers

    Input shape
      - token embedding 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    References
        - [Self-Attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf)
    """

    def __init__(self, head_num=1, dim=50, dropout=0.1, **kwargs):
        super(SASRecBlock, self).__init__(**kwargs)
        self.head_num = head_num
        self.dim = dim
        self.dropout = dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()
        self.attention = MultiHeadSelfAttentionLayer(
            head_num=head_num, key_dim=dim, dropout=dropout
        )
        self.ff = FeedForward(ff_dim=dim, dropout=dropout, model_dim=dim)

    def call(self, inputs, training=False):
        inputs = self.norm(inputs, training=training)
        inputs = self.add(
            [
                inputs,
                self.dropout1(
                    # must enable causal mask
                    self.attention(
                        inputs, inputs, inputs, training=training, use_causal_mask=True
                    ),
                    training=training,
                ),
            ]
        )
        return self.ff(
            inputs,
            training=training,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_num": self.head_num,
                "dim": self.dim,
                "dropout": self.dropout,
            }
        )
        return config


class SASRec(tf.keras.layers.Layer):
    """SASRec model is a stack of self attention layers

    Input shape
      - sequential token index 2D tensor with shape: ``(batch_size, sequence_length)``.
      - positive token index 2D tensor with shape: ``(batch_size, sequence_length)``.
      - negative token index 2D tensor with shape: ``(batch_size, sequence_length)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, sequence_length, 2)``.

    References
        - [Self-Attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf)
    """

    def __init__(
        self,
        vocab_size,
        head_num=1,
        block_num=2,
        seq_length=50,
        dim=50,
        dropout=0.1,
        **kwargs
    ):
        super(SASRec, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.head_num = head_num
        self.block_num = block_num
        self.seq_length = seq_length
        self.dim = dim
        self.dropout = dropout
        # will be reused to general pos and neg embeddings
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=dim, mask_zero=True
        )
        self.positional_emb = PositionalEmbedding(
            self.token_emb, seq_length=seq_length, dim=dim
        )
        self.sas_blocks = [
            SASRecBlock(head_num=head_num, dim=dim, dropout=dropout)
            for _ in range(block_num)
        ]

    def call(self, inputs, training=False):
        input_token, pos, neg = inputs
        # shape [batch_size, token_length, dim]
        input_emb = self.positional_emb(input_token)
        pos_emb = self.token_emb(pos)
        neg_emb = self.token_emb(neg)
        for sas_block in self.sas_blocks:
            output_emb = sas_block(input_emb, training=training)
        # shape [batch_size, token_length, 1]
        pos_logits = tf.reduce_sum(output_emb * pos_emb, axis=-1, keepdims=True)
        neg_logits = tf.reduce_sum(output_emb * neg_emb, axis=-1, keepdims=True)
        # shape [batch_size, token_length, 2]
        return tf.concat([pos_logits, neg_logits], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "block_num": self.block_num,
                "seq_length": self.seq_length,
                "head_num": self.head_num,
                "dim": self.dim,
                "dropout": self.dropout,
            }
        )
        return config
