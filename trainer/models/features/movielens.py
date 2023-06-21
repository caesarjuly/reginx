from typing import Dict
import tensorflow as tf
import tensorflow_recommenders as tfrs

from trainer.util.tools import ObjectDict


class MovieLensQueryEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
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
        return tf.concat(
            [
                self.user_gender(tf.where(inputs["user_gender"], 1, 0)),
                self.user_embedding(inputs["user_id"]),
                self.user_occupation_label(inputs["user_occupation_label"]),
                self.user_zip_code(inputs["user_zip_code"]),
                self.age_embedding(tf.cast(inputs["bucketized_user_age"], tf.int64)),
                self.day_of_week(inputs["day_of_week"]),
                self.hour_of_day(inputs["hour_of_day"]),
                self.ts_cross((inputs["day_of_week"], inputs["hour_of_day"])),
                tf.reshape(inputs["example_age"], [-1, 1]),
                tf.reshape(inputs["example_age_square"], [-1, 1]),
                tf.reshape(inputs["example_age_sqrt"], [-1, 1]),
            ],
            axis=-1,
        )


class MovieLensCandidateEmb(tfrs.Model):
    def __init__(self, meta: Dict):
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
                tf.keras.layers.IntegerLookup(vocabulary=meta["movie_genres"]),
                tf.keras.layers.Embedding(
                    input_dim=len((meta["movie_genres"])),
                    output_dim=64,
                    mask_zero=True,
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )

    def call(self, inputs, training=False):
        return tf.concat(
            [
                self.title_text_embedding(inputs["movie_title"]),
                self.movie_id_embedding(inputs["movie_id"]),
                self.genres(inputs["movie_genres"]),
            ],
            axis=-1,
        )


class MovieLensRankingEmb(tfrs.Model):
    def __init__(self, meta: Dict):
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
                tf.keras.layers.IntegerLookup(vocabulary=meta["movie_genres"]),
                tf.keras.layers.Embedding(
                    input_dim=len((meta["movie_genres"])),
                    output_dim=64,
                    mask_zero=True,
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )

    def call(self, inputs, training=False):
        return tf.concat(
            [
                self.user_gender(tf.where(inputs["user_gender"], 1, 0)),
                self.user_embedding(inputs["user_id"]),
                self.user_occupation_label(inputs["user_occupation_label"]),
                self.user_zip_code(inputs["user_zip_code"]),
                self.age_embedding(tf.cast(inputs["bucketized_user_age"], tf.int64)),
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
        )


class MovieLensWideEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        self.gender_movie_id_cross_layer = tf.keras.layers.StringLookup(
            vocabulary=meta["gender_movie_id_cross"], output_mode="one_hot"
        )
        self.occupation_movie_id_cross_layer = tf.keras.layers.StringLookup(
            vocabulary=meta["occupation_movie_id_cross"], output_mode="one_hot"
        )
        self.zip_movie_id_cross_layer = tf.keras.layers.StringLookup(
            vocabulary=meta["zip_movie_id_cross"], output_mode="one_hot"
        )
        self.age_movie_id_cross_layer = tf.keras.layers.StringLookup(
            vocabulary=meta["age_movie_id_cross"], output_mode="one_hot"
        )

    def call(self, inputs, training=False):
        return tf.concat(
            [
                self.gender_movie_id_cross_layer(
                    tf.strings.join(
                        [
                            tf.strings.as_string(inputs["user_gender"]),
                            inputs["movie_id"],
                        ],
                        separator="_",
                    )
                ),
                self.occupation_movie_id_cross_layer(
                    tf.strings.join(
                        [
                            tf.strings.as_string(inputs["user_occupation_label"]),
                            inputs["movie_id"],
                        ],
                        separator="_",
                    )
                ),
                self.zip_movie_id_cross_layer(
                    tf.strings.join(
                        [inputs["user_zip_code"], inputs["movie_id"]],
                        separator="_",
                    )
                ),
                self.age_movie_id_cross_layer(
                    tf.strings.join(
                        [
                            tf.strings.as_string(inputs["bucketized_user_age"]),
                            inputs["movie_id"],
                        ],
                        separator="_",
                    )
                ),
            ],
            axis=-1,
        )


# FM
class MovieLensFMWideUserEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        # user
        self.user_gender = tf.keras.layers.CategoryEncoding(
            num_tokens=2, output_mode="one_hot"
        )
        self.user_occupation_label = tf.keras.layers.CategoryEncoding(
            num_tokens=22, output_mode="one_hot"
        )
        self.user_zip_code = tf.keras.layers.StringLookup(
            vocabulary=meta["user_zip_code"], output_mode="one_hot"
        )

        self.age_embedding = tf.keras.layers.IntegerLookup(
            vocabulary=meta["bucketized_user_age"], output_mode="one_hot"
        )

        self.day_of_week = tf.keras.layers.CategoryEncoding(
            num_tokens=7, output_mode="one_hot"
        )
        self.hour_of_day = tf.keras.layers.CategoryEncoding(
            num_tokens=24, output_mode="one_hot"
        )

    def call(self, inputs, training=False):
        return tf.concat(
            [
                self.user_gender(tf.where(inputs["user_gender"], 1, 0)),
                self.user_occupation_label(inputs["user_occupation_label"]),
                self.user_zip_code(inputs["user_zip_code"]),
                self.age_embedding(tf.cast(inputs["bucketized_user_age"], tf.int64)),
                self.day_of_week(inputs["day_of_week"]),
                self.hour_of_day(inputs["hour_of_day"]),
            ],
            axis=-1,
        )


class MovieLensFMWideItemEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        # movie
        self.movie_id_embedding = tf.keras.layers.StringLookup(
            vocabulary=meta["movie_id"], output_mode="one_hot"
        )

        self.title_text_embedding = tf.keras.layers.TextVectorization(
            vocabulary=meta["movie_title"], output_mode="multi_hot"
        )

        self.genres = tf.keras.layers.IntegerLookup(
            vocabulary=meta["movie_genres"],
            output_mode="multi_hot",
        )

    def call(self, inputs, training=False):
        return tf.concat(
            [
                self.title_text_embedding(inputs["movie_title"]),
                self.movie_id_embedding(inputs["movie_id"]),
                self.genres(inputs["movie_genres"]),
                tf.reshape(inputs["example_age"], [-1, 1]),
                tf.reshape(inputs["example_age_square"], [-1, 1]),
                tf.reshape(inputs["example_age_sqrt"], [-1, 1]),
            ],
            axis=-1,
        )


class MovieLensFMDeepUserEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        fm_output_dim = 20
        # user
        self.user_gender = tf.keras.layers.Embedding(
            input_dim=2,
            output_dim=fm_output_dim,
        )
        self.user_occupation_label = tf.keras.layers.Embedding(
            input_dim=22,
            output_dim=fm_output_dim,
        )
        self.user_zip_code = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["user_zip_code"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["user_zip_code"]),
                    output_dim=fm_output_dim,
                ),
            ]
        )

        self.age_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=meta["bucketized_user_age"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["bucketized_user_age"]),
                    output_dim=fm_output_dim,
                ),
            ]
        )
        self.user_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["user_id"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["user_id"]), output_dim=fm_output_dim
                ),
            ]
        )
        self.day_of_week = tf.keras.layers.Embedding(
            input_dim=7,
            output_dim=fm_output_dim,
        )
        self.hour_of_day = tf.keras.layers.Embedding(
            input_dim=24,
            output_dim=fm_output_dim,
        )

    def call(self, inputs, training=False):
        return tf.stack(
            [
                self.user_gender(tf.where(inputs["user_gender"], 1, 0)),
                self.user_embedding(inputs["user_id"]),
                self.user_occupation_label(inputs["user_occupation_label"]),
                self.user_zip_code(inputs["user_zip_code"]),
                self.age_embedding(tf.cast(inputs["bucketized_user_age"], tf.int64)),
                self.day_of_week(inputs["day_of_week"]),
                self.hour_of_day(inputs["hour_of_day"]),
            ],
            axis=1,
        )


class MovieLensFMDeepItemEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        fm_output_dim = 20

        # movie
        self.movie_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["movie_id"]),
                tf.keras.layers.Embedding(len(meta["movie_id"]), fm_output_dim),
            ]
        )
        self.title_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(vocabulary=meta["movie_title"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["movie_title"]),
                    output_dim=fm_output_dim,
                    mask_zero=True,
                ),
                # We average the embedding of individual words to get one embedding vector per title.
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )
        self.genres = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=meta["movie_genres"]),
                tf.keras.layers.Embedding(
                    input_dim=len((meta["movie_genres"])),
                    output_dim=fm_output_dim,
                    mask_zero=True,
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )

    def call(self, inputs, training=False):
        return tf.stack(
            [
                self.title_text_embedding(inputs["movie_title"]),
                self.movie_id_embedding(inputs["movie_id"]),
                self.genres(inputs["movie_genres"]),
            ],
            axis=1,
        )


# DeepFM
class MovieLensDeepFMWideEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        # user
        self.user_gender = tf.keras.layers.CategoryEncoding(
            num_tokens=2, output_mode="one_hot"
        )
        self.user_occupation_label = tf.keras.layers.CategoryEncoding(
            num_tokens=22, output_mode="one_hot"
        )
        self.user_zip_code = tf.keras.layers.StringLookup(
            vocabulary=meta["user_zip_code"], output_mode="one_hot"
        )

        self.age_embedding = tf.keras.layers.IntegerLookup(
            vocabulary=meta["bucketized_user_age"], output_mode="one_hot"
        )

        self.day_of_week = tf.keras.layers.CategoryEncoding(
            num_tokens=7, output_mode="one_hot"
        )
        self.hour_of_day = tf.keras.layers.CategoryEncoding(
            num_tokens=24, output_mode="one_hot"
        )

        # movie
        self.movie_id_embedding = tf.keras.layers.StringLookup(
            vocabulary=meta["movie_id"], output_mode="one_hot"
        )

        self.title_text_embedding = tf.keras.layers.TextVectorization(
            vocabulary=meta["movie_title"], output_mode="multi_hot"
        )

        self.genres = tf.keras.layers.IntegerLookup(
            vocabulary=meta["movie_genres"],
            output_mode="multi_hot",
        )

    def call(self, inputs, training=False):
        return tf.concat(
            [
                self.user_gender(tf.where(inputs["user_gender"], 1, 0)),
                self.user_occupation_label(inputs["user_occupation_label"]),
                self.user_zip_code(inputs["user_zip_code"]),
                self.age_embedding(tf.cast(inputs["bucketized_user_age"], tf.int64)),
                self.day_of_week(inputs["day_of_week"]),
                self.hour_of_day(inputs["hour_of_day"]),
                self.title_text_embedding(inputs["movie_title"]),
                self.movie_id_embedding(inputs["movie_id"]),
                self.genres(inputs["movie_genres"]),
                tf.reshape(inputs["example_age"], [-1, 1]),
                tf.reshape(inputs["example_age_square"], [-1, 1]),
                tf.reshape(inputs["example_age_sqrt"], [-1, 1]),
            ],
            axis=-1,
        )


class MovieLensDeepFMDeepEmb(tfrs.Model):
    def __init__(self, meta: Dict):
        super().__init__()
        fm_output_dim = 20
        # user
        self.user_gender = tf.keras.layers.Embedding(
            input_dim=2,
            output_dim=fm_output_dim,
        )
        self.user_occupation_label = tf.keras.layers.Embedding(
            input_dim=22,
            output_dim=fm_output_dim,
        )
        self.user_zip_code = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["user_zip_code"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["user_zip_code"]),
                    output_dim=fm_output_dim,
                ),
            ]
        )

        self.age_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=meta["bucketized_user_age"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["bucketized_user_age"]),
                    output_dim=fm_output_dim,
                ),
            ]
        )
        self.user_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["user_id"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["user_id"]), output_dim=fm_output_dim
                ),
            ]
        )
        self.day_of_week = tf.keras.layers.Embedding(
            input_dim=7,
            output_dim=fm_output_dim,
        )
        self.hour_of_day = tf.keras.layers.Embedding(
            input_dim=24,
            output_dim=fm_output_dim,
        )

        # movie
        self.movie_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["movie_id"]),
                tf.keras.layers.Embedding(len(meta["movie_id"]), fm_output_dim),
            ]
        )
        self.title_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(vocabulary=meta["movie_title"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["movie_title"]),
                    output_dim=fm_output_dim,
                    mask_zero=True,
                ),
                # We average the embedding of individual words to get one embedding vector per title.
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )
        self.genres = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=meta["movie_genres"]),
                tf.keras.layers.Embedding(
                    input_dim=len((meta["movie_genres"])),
                    output_dim=fm_output_dim,
                    mask_zero=True,
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )

    def call(self, inputs, training=False):
        return tf.stack(
            [
                self.user_gender(tf.where(inputs["user_gender"], 1, 0)),
                self.user_embedding(inputs["user_id"]),
                self.user_occupation_label(inputs["user_occupation_label"]),
                self.user_zip_code(inputs["user_zip_code"]),
                self.age_embedding(tf.cast(inputs["bucketized_user_age"], tf.int64)),
                self.day_of_week(inputs["day_of_week"]),
                self.hour_of_day(inputs["hour_of_day"]),
                self.title_text_embedding(inputs["movie_title"]),
                self.movie_id_embedding(inputs["movie_id"]),
                self.genres(inputs["movie_genres"]),
            ],
            axis=1,
        )
