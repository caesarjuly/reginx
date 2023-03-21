import tensorflow as tf


def test(model_dir):
    # Load it back; can also be done in TensorFlow Serving.
    loaded = tf.saved_model.load(model_dir)
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
            "example_age": tf.constant(0.5, shape=(1)),
            "example_age_square": tf.constant(0.5, shape=(1)),
            "example_age_sqrt": tf.constant(0.5, shape=(1)),
        }
    )
    print(scores, ids)
