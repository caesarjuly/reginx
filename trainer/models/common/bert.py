import tensorflow as tf

from trainer.models.common.transformer import Encoder


class BertEmbedding(tf.keras.layers.Layer):
    """Bert embedding is composed of a positional embedding layer, a segment embedding layer and a normal embedding layer

    Input shape
      - token index 2D tensor with shape: ``(batch_size, sequence_length)``.
      - segment index 2D tensor with shape: ``(batch_size, sequence_length)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    References
        - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
    """

    def __init__(self, vocab_size, seq_length=128, dim=512, **kwargs):
        super(BertEmbedding, self).__init__(**kwargs)
        assert seq_length % 2 == 0, "Output dimension needs to be an even integer"
        self.length = seq_length
        self.dim = dim
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=dim, mask_zero=True
        )
        self.position_emb = tf.keras.layers.Embedding(
            input_dim=seq_length, output_dim=dim
        )
        self.segment_emb = tf.keras.layers.Embedding(input_dim=2, output_dim=dim)

    def call(self, inputs, **kwargs):
        tokens = inputs["input_word_ids"]
        segments = inputs["input_type_ids"]
        length = tf.shape(tokens)[1]
        embedded_tokens = self.token_emb(tokens)
        embedded_segments = self.segment_emb(segments)
        embedded_positions = self.position_emb(tf.range(length))
        # This factor sets the relative scale of the embedding and positonal_encoding.
        embedded_tokens *= tf.math.sqrt(tf.cast(self.dim, tf.float32))
        return (
            embedded_tokens + embedded_segments + embedded_positions[tf.newaxis, :, :]
        )

    # Pass mask from token_emb, https://www.tensorflow.org/guide/keras/understanding_masking_and_padding#supporting_masking_in_your_custom_layers
    def compute_mask(self, inputs, mask=None):
        return inputs["input_mask"]


class Bert(tf.keras.layers.Layer):
    """Bert model is a stack of multiple layers of encoders

    Input shape
      - 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    References
        - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
    """

    def __init__(
        self,
        vocab_size,
        seq_length,
        layer_num=6,
        model_dim=512,
        ff_dim=2048,
        dropout=0.1,
        head_num=8,
        **kwargs,
    ):
        super(Bert, self).__init__(**kwargs)
        self.emb = BertEmbedding(
            vocab_size=vocab_size, seq_length=seq_length, dim=model_dim
        )
        self.encoders = [
            Encoder(
                model_dim=model_dim, ff_dim=ff_dim, dropout=dropout, head_num=head_num
            )
            for _ in range(layer_num)
        ]

    def call(self, inputs, training=False):
        # shape [batch_size, token_length]
        emb = self.emb(inputs, training=training)
        # shape [batch_size, token_length, model_dim]
        for encoder in self.encoders:
            emb = encoder(emb, training=training)
        # shape [batch_size, token_length, model_dim]
        return emb


class BertMLM(tf.keras.layers.Layer):
    """Masked language model simply mask some percentage of the input tokens at random, and then predict those masked tokens

    Input shape
      - 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    References
        - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
    """

    def __init__(
        self,
        vocab_size,
        seq_length,
        layer_num=6,
        model_dim=512,
        ff_dim=2048,
        dropout=0.1,
        head_num=8,
        **kwargs,
    ):
        super(BertMLM, self).__init__(**kwargs)

        self.emb = Bert(
            vocab_size=vocab_size,
            seq_length=seq_length,
            layer_num=layer_num,
            model_dim=model_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            head_num=head_num,
            **kwargs,
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(
            vocab_size, activation=tf.keras.activations.softmax
        )

    def call(self, inputs, training=False):
        # shape [batch_size, token_length, model_dim]
        emb = self.dropout(self.emb(inputs, training=training))
        # shape [batch_size, token_length, vocab_size_logits]
        emb = self.dense(emb)
        # gather the corresponding logits per the masked_lm_positions
        return tf.gather(emb, inputs["masked_lm_positions"], axis=1, batch_dims=1)


class BertNSP(tf.keras.layers.Layer):
    """Next sentence predition use the CLS token embedding for prediction

    Input shape
      - 3D tensor with shape: ``(batch_size, sequence_length, embedding_size)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, 1)``.

    References
        - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
    """

    def __init__(
        self,
        vocab_size,
        seq_length,
        layer_num=6,
        model_dim=512,
        ff_dim=2048,
        dropout=0.1,
        head_num=8,
        **kwargs,
    ):
        super(BertNSP, self).__init__(**kwargs)

        self.emb = Bert(
            vocab_size=vocab_size,
            seq_length=seq_length,
            layer_num=layer_num,
            model_dim=model_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            head_num=head_num,
            **kwargs,
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=False):
        # shape [batch_size, token_length, model_dim]
        emb = self.dropout(self.emb(inputs, training=training))
        # shape [batch_size, 2]
        return self.dense(emb[:, 0, :])
