from typing import Dict
import tensorflow as tf
import tensorflow_recommenders as tfrs


class CriteoDenseEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        self.total_dense = 13
        self.norms = {
            f"dense_{i}": tf.keras.layers.Normalization(
                mean=meta[f"dense_{i}"]["mean"], variance=meta[f"dense_{i}"]["std"]
            )
            for i in range(1, self.total_dense + 1)
        }

    def call(self, inputs, feature_names=None, training=False):
        if feature_names:
            return tf.concat(
                [
                    self.norms[name](tf.reshape(inputs[name], [-1, 1]))
                    for name in feature_names
                ],
                axis=-1,
            )
        else:
            return tf.concat(
                [
                    self.norms[f"dense_{i}"](tf.reshape(inputs[f"dense_{i}"], [-1, 1]))
                    for i in range(1, self.total_dense + 1)
                ],
                axis=-1,
            )


class CriteoDenseAsWeightEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        self.total_dense = 13
        self.norms = [
            tf.keras.layers.Normalization(
                mean=meta[f"dense_{i}"]["mean"], variance=meta[f"dense_{i}"]["std"]
            )
            for i in range(1, self.total_dense + 1)
        ]
        # shape [feature_num, emb_size]
        self.dense_emb = self.add_weight(
            name="dense_emb",
            shape=[self.total_dense, 16],
        )

    def call(self, inputs, training=False):
        # shape [batch_size, feature_num, 1]
        dense_weights = tf.stack(
            [
                self.norms[i - 1](tf.reshape(inputs[f"dense_{i}"], [-1, 1]))
                for i in range(1, self.total_dense + 1)
            ],
            axis=1,
        )
        # shape [batch_size, feature_num, emb_size]
        return dense_weights * self.dense_emb


# Used as the linear weights for wide part
class CriteoSparseLinearEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        self.total_sparse = 26
        self.embs = [
            tf.keras.Sequential(
                [
                    tf.keras.layers.Hashing(
                        num_bins=meta[f"sparse_{i}"]["unique"] // 5
                    ),
                    tf.keras.layers.Embedding(meta[f"sparse_{i}"]["unique"] // 5, 1),
                ]
            )
            if meta[f"sparse_{i}"]["unique"] > 5_000_000
            else tf.keras.Sequential(
                [
                    tf.keras.layers.StringLookup(
                        vocabulary=meta[f"sparse_{i}"]["vocab"]
                    ),
                    tf.keras.layers.Embedding(meta[f"sparse_{i}"]["unique"], 1),
                ]
            )
            for i in range(1, self.total_sparse + 1)
        ]

    def call(self, inputs, training=False):
        return tf.concat(
            [
                self.embs[i - 1](inputs[f"sparse_{i}"])
                for i in range(1, self.total_sparse + 1)
            ],
            axis=-1,
        )


class CriteoSparseEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        self.total_sparse = 26
        self.embs = {
            f"sparse_{i}": tf.keras.Sequential(
                [
                    tf.keras.layers.Hashing(
                        num_bins=meta[f"sparse_{i}"]["unique"] // 5
                    ),
                    tf.keras.layers.Embedding(meta[f"sparse_{i}"]["unique"] // 5, 16),
                ]
            )
            if meta[f"sparse_{i}"]["unique"] > 5_000_000
            else tf.keras.Sequential(
                [
                    tf.keras.layers.StringLookup(
                        vocabulary=meta[f"sparse_{i}"]["vocab"]
                    ),
                    tf.keras.layers.Embedding(meta[f"sparse_{i}"]["unique"], 16),
                ]
            )
            for i in range(1, self.total_sparse + 1)
        }

    def call(self, inputs, feature_names=None, training=False):
        if feature_names:
            return tf.stack(
                [self.embs[name](inputs[name]) for name in feature_names],
                axis=1,
            )
        else:
            return tf.stack(
                [
                    self.embs[f"sparse_{i}"](inputs[f"sparse_{i}"])
                    for i in range(1, self.total_sparse + 1)
                ],
                axis=1,
            )


class CriteoRankingEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        self.dense_emb = CriteoDenseEmb(meta)
        self.sparse_emb = CriteoSparseEmb(meta)

    def call(self, inputs, training=False):
        dense_emb = self.dense_emb(inputs, training=training)
        sparse_emb = self.sparse_emb(inputs, training=training)
        concat_sparse_emb = tf.concat(tf.unstack(sparse_emb, axis=1), axis=-1)
        return tf.concat([dense_emb, concat_sparse_emb], axis=-1)


class CriteoDenseAsWeightRankingEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        self.dense_emb = CriteoDenseAsWeightEmb(meta)
        self.sparse_emb = CriteoSparseEmb(meta)

    def call(self, inputs, training=False):
        dense_emb = self.dense_emb(inputs, training=training)
        sparse_emb = self.sparse_emb(inputs, training=training)
        return tf.concat([dense_emb, sparse_emb], axis=1)
