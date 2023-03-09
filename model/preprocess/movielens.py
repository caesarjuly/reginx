import tensorflow as tf
import tensorflow_datasets as tfds
import json

from model.util.json import NpEncoder


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

    movie_id_layer = tf.keras.layers.StringLookup()
    movie_id_layer.adapt(movies.map(lambda x: x["movie_id"]))
    # -1 will be the first token as the masking value and oov value
    movie_genres_layer = tf.keras.layers.IntegerLookup(mask_token=-2)
    movie_genres_layer.adapt(movies.map(lambda x: x["movie_genres"]))
    movie_title_layer = tf.keras.layers.TextVectorization()
    movie_title_layer.adapt(movies.map(lambda x: x["movie_title"]))
    meta_dict = {
        "user_id": user_id_layer.get_vocabulary(),
        "user_zip_code": user_zip_code_layer.get_vocabulary(),
        "bucketized_user_age": bucketized_user_age_layer.get_vocabulary(),
        "movie_id": movie_id_layer.get_vocabulary(),
        "movie_genres": movie_genres_layer.get_vocabulary(),
        "movie_title": movie_title_layer.get_vocabulary(),
    }
    with open("model/meta/movie_lens.json", "w+") as fp:
        json.dump(meta_dict, fp, cls=NpEncoder)


if __name__ == "__main__":
    build_meta()
