import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import numpy as np

from trainer.util.json import NpEncoder
from trainer.common.gcp import upload_from_directory, BUCKET_NAME

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def build_meta() -> None:
    # Ratings data.
    ratings = tfds.load("movielens/1m-ratings", split="train")
    # Features of all the available movies.
    movies = tfds.load("movielens/1m-movies", split="train")
    ratings = ratings.map(
        lambda x: {
            "movie_id": x["movie_id"],
            "user_id": x["user_id"],
            "user_gender": x["user_gender"],
            "user_occupation_label": x["user_occupation_label"],
            "user_zip_code": x["user_zip_code"],
            "bucketized_user_age": x["bucketized_user_age"],
        }
    )

    movies = movies.map(
        lambda x: {
            "movie_id": x["movie_id"],
            "movie_title": x["movie_title"],
            "movie_genres": x["movie_genres"],
        }
    )

    user_id_layer = tf.keras.layers.StringLookup()
    user_id_layer.adapt(ratings.map(lambda x: x["user_id"]))
    user_zip_code_layer = tf.keras.layers.StringLookup()
    user_zip_code_layer.adapt(ratings.map(lambda x: x["user_zip_code"]))
    bucketized_user_age_layer = tf.keras.layers.IntegerLookup()
    bucketized_user_age_layer.adapt(
        ratings.map(lambda x: int(x["bucketized_user_age"]))
    )

    gender_movie_id_cross_layer = tf.keras.layers.StringLookup()
    gender_movie_id_cross_layer.adapt(
        ratings.map(
            lambda x: tf.strings.join(
                [tf.strings.as_string(x["user_gender"]), x["movie_id"]],
                separator="_",
            )
        )
    )
    occupation_movie_id_cross_layer = tf.keras.layers.StringLookup()
    occupation_movie_id_cross_layer.adapt(
        ratings.map(
            lambda x: tf.strings.join(
                [tf.strings.as_string(x["user_occupation_label"]), x["movie_id"]],
                separator="_",
            )
        )
    )
    zip_movie_id_cross_layer = tf.keras.layers.StringLookup()
    zip_movie_id_cross_layer.adapt(
        ratings.map(
            lambda x: tf.strings.join(
                [x["user_zip_code"], x["movie_id"]],
                separator="_",
            )
        )
    )
    age_movie_id_cross_layer = tf.keras.layers.StringLookup()
    age_movie_id_cross_layer.adapt(
        ratings.map(
            lambda x: tf.strings.join(
                [tf.strings.as_string(x["bucketized_user_age"]), x["movie_id"]],
                separator="_",
            )
        )
    )

    movie_id_layer = tf.keras.layers.StringLookup()
    movie_id_layer.adapt(movies.map(lambda x: x["movie_id"]))
    # -1 will be the first token as the masking value and oov value
    movie_genres_layer = tf.keras.layers.IntegerLookup()
    movie_genres_layer.adapt(movies.map(lambda x: x["movie_genres"]))
    movie_title_layer = tf.keras.layers.TextVectorization()
    movie_title_layer.adapt(movies.map(lambda x: x["movie_title"]))
    meta_dict = {
        "user_id": user_id_layer.get_vocabulary(),
        "user_zip_code": user_zip_code_layer.get_vocabulary(),
        "bucketized_user_age": bucketized_user_age_layer.get_vocabulary(),
        "gender_movie_id_cross": gender_movie_id_cross_layer.get_vocabulary(),
        "occupation_movie_id_cross": occupation_movie_id_cross_layer.get_vocabulary(),
        "zip_movie_id_cross": zip_movie_id_cross_layer.get_vocabulary(),
        "age_movie_id_cross": age_movie_id_cross_layer.get_vocabulary(),
        "movie_id": movie_id_layer.get_vocabulary(),
        "movie_genres": movie_genres_layer.get_vocabulary(),
        "movie_title": movie_title_layer.get_vocabulary(),
    }
    with open("trainer/meta/movie_lens.json", "w+") as fp:
        json.dump(meta_dict, fp, cls=NpEncoder)


def convert_to_hour_of_day(ts: int):
    return datetime.datetime.fromtimestamp(ts).hour


def convert_to_day_of_week(ts: int):
    return datetime.datetime.fromtimestamp(ts).weekday()


def transform_data():
    # Ratings data.
    ratings = tfds.load("movielens/1m-ratings", split="train")
    # Features of all the available movies.
    movies = tfds.load("movielens/1m-movies", split="train")

    max_ts = np.max(list(ratings.map(lambda x: x["timestamp"])))
    min_ts = np.min(list(ratings.map(lambda x: x["timestamp"])))
    gap = max_ts - min_ts

    ratings_train, ratings_test = ratings.take(900_000), ratings.skip(900_000).take(
        100_000
    )

    # Select the basic features.
    ratings_train = ratings_train.map(
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
            "example_age": (x["timestamp"] - min_ts) / gap,
            "example_age_square": tf.math.square((x["timestamp"] - min_ts) / gap),
            "example_age_sqrt": tf.math.sqrt((x["timestamp"] - min_ts) / gap),
        }
    )
    ratings_test = ratings_test.map(
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
            "example_age": 0.0,
            "example_age_square": 0.0,
            "example_age_sqrt": 0.0,
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
    uniform_negatives = movies.cache().repeat().shuffle(1_000)

    train_negatives = tf.data.Dataset.zip((ratings_train, uniform_negatives)).map(
        lambda x, y: {
            "movie_id": y["movie_id"],
            "user_id": x["user_id"],
            "user_gender": x["user_gender"],
            "user_occupation_label": x["user_occupation_label"],
            "user_zip_code": x["user_zip_code"],
            "bucketized_user_age": x["bucketized_user_age"],
            "movie_genres": y["movie_genres"],
            "movie_title": y["movie_title"],
            "day_of_week": x["day_of_week"],
            "hour_of_day": x["hour_of_day"],
            "user_rating": 0.0,
            "example_age": x["example_age"],
            "example_age_square": x["example_age_square"],
            "example_age_sqrt": x["example_age_sqrt"],
        }
    )

    train_with_negatives = tf.data.Dataset.sample_from_datasets(
        [ratings_train, train_negatives], weights=[0.5, 0.5]
    )
    test_negatives = tf.data.Dataset.zip((ratings_test, uniform_negatives)).map(
        lambda x, y: {
            "movie_id": y["movie_id"],
            "user_id": x["user_id"],
            "user_gender": x["user_gender"],
            "user_occupation_label": x["user_occupation_label"],
            "user_zip_code": x["user_zip_code"],
            "bucketized_user_age": x["bucketized_user_age"],
            "movie_genres": y["movie_genres"],
            "movie_title": y["movie_title"],
            "day_of_week": x["day_of_week"],
            "hour_of_day": x["hour_of_day"],
            "user_rating": 0.0,
            "example_age": x["example_age"],
            "example_age_square": x["example_age_square"],
            "example_age_sqrt": x["example_age_sqrt"],
        }
    )
    test_with_negatives = tf.data.Dataset.sample_from_datasets(
        [ratings_test, test_negatives], weights=[0.5, 0.5]
    )

    ratings_train.shuffle(1_000).save("ratings_train")
    ratings_test.shuffle(1_000).save("ratings_test")
    # follow the practice from DCN v2, ignore rating 3
    ratings_train.filter(lambda x: x["user_rating"] != 3).shuffle(1_000).save(
        "ratings_train_remove_rating3"
    )
    ratings_test.filter(lambda x: x["user_rating"] != 3).shuffle(1_000).save(
        "ratings_test_remove_rating3"
    )
    train_with_negatives.shuffle(1_000).save("ratings_train_random_neg")
    test_with_negatives.shuffle(1_000).save("ratings_test_random_neg")
    movies.save("movies")
    upload_from_directory("ratings_train", BUCKET_NAME, "movielens/data/ratings_train")
    upload_from_directory("ratings_test", BUCKET_NAME, "movielens/data/ratings_test")
    upload_from_directory(
        "ratings_train_remove_rating3",
        BUCKET_NAME,
        "movielens/data/ratings_train_remove_rating3",
    )
    upload_from_directory(
        "ratings_test_remove_rating3",
        BUCKET_NAME,
        "movielens/data/ratings_test_remove_rating3",
    )
    upload_from_directory(
        "ratings_train_random_neg",
        BUCKET_NAME,
        "movielens/data/ratings_train_random_neg",
    )
    upload_from_directory(
        "ratings_test_random_neg", BUCKET_NAME, "movielens/data/ratings_test_random_neg"
    )
    upload_from_directory("movies", BUCKET_NAME, "movielens/data/movies")


if __name__ == "__main__":
    # build_meta()
    transform_data()
