from typing import Dict
import tensorflow as tf

from trainer.util.tools import ObjectDict


class MovieLensQueryModel(tf.keras.Model):
    def __init__(self, hparams: ObjectDict, meta: Dict):
        super().__init__()
        self.hparams = hparams
        self.user_gender = tf.keras.layers.CategoryEncoding(
            num_tokens=2, output_mode="one_hot"
        )
        self.user_occupation_label = tf.keras.layers.CategoryEncoding(
            num_tokens=22, output_mode="one_hot"
        )
        self.user_zip_code = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["user_zip_code"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["user_zip_code"]),
                    output_dim=64,
                ),
            ]
        )

        self.age_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=meta["bucketized_user_age"]),
                tf.keras.layers.CategoryEncoding(
                    num_tokens=len(meta["bucketized_user_age"]),
                    output_mode="one_hot",
                ),
            ]
        )
        self.user_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["user_id"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["user_id"]), output_dim=128
                ),
            ]
        )
        self.day_of_week = tf.keras.layers.CategoryEncoding(
            num_tokens=7, output_mode="one_hot"
        )
        self.hour_of_day = tf.keras.layers.CategoryEncoding(
            num_tokens=24, output_mode="one_hot"
        )
        self.ts_cross = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.HashedCrossing(num_bins=34),
                tf.keras.layers.Embedding(input_dim=35, output_dim=32),
            ]
        )
        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
            ]
        )

    def call(self, inputs, training=False):
        user_embeddings = self.dense(
            tf.concat(
                [
                    self.user_gender(tf.where(inputs["user_gender"], 1, 0)),
                    self.user_embedding(inputs["user_id"]),
                    self.user_occupation_label(inputs["user_occupation_label"]),
                    self.user_zip_code(inputs["user_zip_code"]),
                    self.age_embedding(
                        tf.cast(inputs["bucketized_user_age"], tf.int64)
                    ),
                    self.day_of_week(inputs["day_of_week"]),
                    self.hour_of_day(inputs["hour_of_day"]),
                    self.ts_cross((inputs["day_of_week"], inputs["hour_of_day"])),
                    tf.reshape(inputs["example_age"], [-1, 1]),
                    tf.reshape(inputs["example_age_square"], [-1, 1]),
                    tf.reshape(inputs["example_age_sqrt"], [-1, 1]),
                ],
                axis=-1,
            ),
            training,
        )
        return tf.math.l2_normalize(user_embeddings, -1)


class MovieLensCandidateModel(tf.keras.Model):
    def __init__(self, hparams: ObjectDict, meta: Dict):
        super().__init__()
        self.movie_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["movie_id"]),
                tf.keras.layers.Embedding(len(meta["movie_id"]), 128),
            ]
        )
        self.title_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(vocabulary=meta["movie_title"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["movie_title"]),
                    output_dim=128,
                    mask_zero=True,
                ),
                # We average the embedding of individual words to get one embedding vector per title.
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )
        self.genres = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(
                    mask_token=-2, vocabulary=meta["movie_genres"]
                ),
                tf.keras.layers.Embedding(
                    input_dim=len((meta["movie_genres"])),
                    output_dim=64,
                    mask_zero=True,
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )
        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
            ]
        )

    def call(self, inputs, training=False):
        movie_embeddings = self.dense(
            tf.concat(
                [
                    self.title_text_embedding(inputs["movie_title"]),
                    self.movie_id_embedding(inputs["movie_id"]),
                    self.genres(inputs["movie_genres"]),
                ],
                axis=-1,
            ),
            training,
        )
        return tf.math.l2_normalize(movie_embeddings, -1)


class MovieLensModel(tf.keras.Model):
    def __init__(self, hparams: ObjectDict, meta: Dict):
        super().__init__()
        # user
        self.user_gender = tf.keras.layers.CategoryEncoding(
            num_tokens=2, output_mode="one_hot"
        )
        self.user_occupation_label = tf.keras.layers.CategoryEncoding(
            num_tokens=22, output_mode="one_hot"
        )
        self.user_zip_code = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["user_zip_code"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["user_zip_code"]),
                    output_dim=64,
                ),
            ]
        )

        self.age_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=meta["bucketized_user_age"]),
                tf.keras.layers.CategoryEncoding(
                    num_tokens=len(meta["bucketized_user_age"]),
                    output_mode="one_hot",
                ),
            ]
        )
        self.user_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["user_id"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["user_id"]), output_dim=128
                ),
            ]
        )
        self.day_of_week = tf.keras.layers.CategoryEncoding(
            num_tokens=7, output_mode="one_hot"
        )
        self.hour_of_day = tf.keras.layers.CategoryEncoding(
            num_tokens=24, output_mode="one_hot"
        )
        self.ts_cross = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.HashedCrossing(num_bins=34),
                tf.keras.layers.Embedding(input_dim=35, output_dim=32),
            ]
        )

        # movie
        self.movie_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["movie_id"]),
                tf.keras.layers.Embedding(len(meta["movie_id"]), 128),
            ]
        )
        self.title_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(vocabulary=meta["movie_title"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["movie_title"]),
                    output_dim=128,
                    mask_zero=True,
                ),
                # We average the embedding of individual words to get one embedding vector per title.
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )
        self.genres = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(
                    mask_token=-2, vocabulary=meta["movie_genres"]
                ),
                tf.keras.layers.Embedding(
                    input_dim=len((meta["movie_genres"])),
                    output_dim=64,
                    mask_zero=True,
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )
        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(1),
            ]
        )
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def call(self, inputs, training=False):
        prediction = self.dense(
            tf.concat(
                [
                    self.user_gender(tf.where(inputs["user_gender"], 1, 0)),
                    self.user_embedding(inputs["user_id"]),
                    self.user_occupation_label(inputs["user_occupation_label"]),
                    self.user_zip_code(inputs["user_zip_code"]),
                    self.age_embedding(
                        tf.cast(inputs["bucketized_user_age"], tf.int64)
                    ),
                    self.day_of_week(inputs["day_of_week"]),
                    self.hour_of_day(inputs["hour_of_day"]),
                    self.ts_cross((inputs["day_of_week"], inputs["hour_of_day"])),
                    tf.reshape(inputs["example_age"], [-1, 1]),
                    tf.reshape(inputs["example_age_square"], [-1, 1]),
                    tf.reshape(inputs["example_age_sqrt"], [-1, 1]),
                    self.title_text_embedding(inputs["movie_title"]),
                    self.movie_id_embedding(inputs["movie_id"]),
                    self.genres(inputs["movie_genres"]),
                ],
                axis=-1,
            ),
            training,
        )
        logits = self.sigmoid(prediction)
        return logits
