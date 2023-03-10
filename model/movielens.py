import json
from typing import Dict, Text, Tuple
import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from model.common.sampling_bias_correction import SamplingBiasCorrection


def convert_to_hour_of_day(ts: int):
    return datetime.datetime.fromtimestamp(ts).hour


def convert_to_day_of_week(ts: int):
    return datetime.datetime.fromtimestamp(ts).weekday()


def load_meta():
    with open("model/meta/movie_lens.json", "r") as f:
        return json.load(f)


def load_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    # Ratings data.
    ratings = tfds.load("movielens/1m-ratings", split="train")
    # Features of all the available movies.
    movies = tfds.load("movielens/1m-movies", split="train")

    # Select the basic features.
    ratings = ratings.map(
        lambda x: {
            "movie_id": x["movie_id"],
            "user_id": x["user_id"],
            "user_gender": x["user_gender"],
            "user_occupation_label": x["user_occupation_label"],
            "user_zip_code": x["user_zip_code"],
            "bucketized_user_age": x["bucketized_user_age"],
            "movie_genres": tf.pad(
                x["movie_genres"],
                [[0, 6 - tf.shape(x["movie_genres"])[0]]],
                constant_values=-2,
            ),
            "movie_title": x["movie_title"],
            "day_of_week": tf.reshape(
                tf.py_function(convert_to_day_of_week, [x["timestamp"]], tf.int64), []
            ),
            "hour_of_day": tf.reshape(
                tf.py_function(convert_to_hour_of_day, [x["timestamp"]], tf.int64), []
            ),
            "user_rating": x["user_rating"],
        }
    )

    movies = movies.map(
        lambda x: {
            "movie_id": x["movie_id"],
            "movie_title": x["movie_title"],
            "movie_genres": tf.pad(
                x["movie_genres"],
                [[0, 6 - tf.shape(x["movie_genres"])[0]]],
                constant_values=-2,
            ),
        }
    )
    return ratings, movies


class UserModel(tf.keras.Model):
    def __init__(self, meta):
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
                    output_dim=32,
                ),
            ]
        )

        self.age_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.IntegerLookup(vocabulary=meta["bucketized_user_age"]),
                #           tf.keras.layers.Embedding(len(age_buckets) + 1, 32)
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
                    input_dim=len(meta["user_id"]), output_dim=64
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
                tf.keras.layers.Embedding(input_dim=35, output_dim=16),
            ]
        )
        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
            ]
        )

    def call(self, inputs):
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
                ],
                axis=-1,
            )
        )
        return tf.math.l2_normalize(user_embeddings, -1)


class MovieModel(tf.keras.Model):
    def __init__(self, meta):
        super().__init__()
        self.movie_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=meta["movie_id"]),
                tf.keras.layers.Embedding(len(meta["movie_id"]), 64),
            ]
        )
        self.title_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(vocabulary=meta["movie_title"]),
                tf.keras.layers.Embedding(
                    input_dim=len(meta["movie_title"]),
                    output_dim=64,
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
                    output_dim=32,
                    mask_zero=True,
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )
        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
            ]
        )

    def call(self, inputs):
        movie_embeddings = self.dense(
            tf.concat(
                [
                    self.title_text_embedding(inputs["movie_title"]),
                    self.movie_id_embedding(inputs["movie_id"]),
                    self.genres(inputs["movie_genres"]),
                ],
                axis=-1,
            )
        )
        return tf.math.l2_normalize(movie_embeddings, -1)


class Model(tfrs.Model):
    def __init__(self, movies: tf.data.Dataset, meta: Dict):
        super().__init__()
        self.query_model = UserModel(meta)
        self.candidate_model = MovieModel(meta)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(
                    lambda x: (x["movie_id"], self.candidate_model(x))
                ),
            ),
            remove_accidental_hits=True,
            temperature=0.05,
        )
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.sampling_bias = SamplingBiasCorrection()

    def compute_loss(self, inputs, training=False) -> tf.Tensor:
        if training:
            self.global_step.assign_add(1)
            features, extra_items = inputs
            user_embeddings = self.query_model(features)
            candidates_embeddings = self.candidate_model(features)
            negatives_embeddings = self.candidate_model(extra_items)
            # we cannot turn on the topK metrics calculation for training when there is extra negatives, need modification on the Retrieval call function
            # true_candidate_ids=candidate_ids[:tf.shape(query_embeddings)[0]])
            candidate_ids = tf.concat(
                [features["movie_id"], extra_items["movie_id"]], axis=-1
            )
            candidate_embeddings = tf.concat(
                [candidates_embeddings, negatives_embeddings], axis=0
            )
            candidate_sampling_probability = self.sampling_bias(
                self.global_step, candidate_ids
            )
        else:
            user_embeddings = self.query_model(inputs)
            candidate_embeddings = self.candidate_model(inputs)
            candidate_sampling_probability = None
            candidate_ids = inputs["movie_id"]
        return self.task(
            user_embeddings,
            candidate_embeddings,
            candidate_sampling_probability=candidate_sampling_probability,
            candidate_ids=candidate_ids,
            compute_metrics=not training,
        )


def train(ratings: tf.data.Dataset, movies: tf.data.Dataset, meta: Dict) -> None:

    model = Model(movies, meta)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.05))
    # Randomly shuffle data and split between train and test.
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(10_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(900_000).batch(256).cache()
    test = shuffled.skip(900_000).take(100_000).batch(256).cache()
    uniform_negatives = movies.cache().repeat().shuffle(1_000).batch(128)
    train_with_mns = tf.data.Dataset.zip((train, uniform_negatives))

    # Train.
    model.fit(train_with_mns, epochs=1)
    # evaluate
    model.evaluate(test, return_dict=True)
    return model


def save_model(model: tf.keras.Model, movies: tf.data.Dataset) -> None:

    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
        movies.batch(128).map(lambda x: (x["movie_id"], model.candidate_model(x)))
    )
    # https://github.com/tensorflow/tensorflow/issues/37439#issuecomment-596916472
    index.predict(ratings.take(10).batch(5))
    # Save the index.
    # https://github.com/tensorflow/models/issues/8990#issuecomment-1069733488
    tf.saved_model.save(index, "model/saved_models/movielens")

    # Load it back; can also be done in TensorFlow Serving.
    loaded = tf.saved_model.load("model/saved_models/movielens")
    print(list(loaded.signatures.keys()))

    # Pass a user id in, get top predicted movie titles back.
    scores, ids = loaded(
        {
            "bucketized_user_age": tf.constant(35.0, shape=(1)),
            "movie_genres": tf.constant(
                [0, 7, -2, -2, -2, -2], shape=(1, 6), dtype=tf.int64
            ),
            "movie_id": tf.constant("3107", shape=(1)),
            "movie_title": tf.constant("Backdraft (1991)", shape=(1)),
            "day_of_week": tf.constant(3, shape=(1), dtype=tf.int64),
            "hour_of_day": tf.constant(20, shape=(1), dtype=tf.int64),
            "user_gender": tf.constant(True, shape=(1)),
            "user_id": tf.constant("130", shape=(1)),
            "user_occupation_label": tf.constant(18, shape=(1), dtype=tf.int64),
            "user_rating": tf.constant(5.0, shape=(1)),
            "user_zip_code": tf.constant("50021", shape=(1)),
        }
    )
    print(scores, ids)


if __name__ == "__main__":
    meta = load_meta()
    ratings, movies = load_data()
    model = train(ratings, movies, meta)
    save_model(model, movies)
