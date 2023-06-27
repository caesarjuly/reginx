from trainer.common.gcp import BUCKET_NAME, download_from_directory
from trainer.tasks.base import BaseTask
from trainer.models.common.feature_cross import FMLayer
import tensorflow as tf


class GenerateFMEmbedding(BaseTask):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)

    def load_data(self) -> tf.data.Dataset:
        download_from_directory(BUCKET_NAME, self.hparams.data, "/tmp/train")
        return tf.data.Dataset.load("/tmp/train")

    def run(self):
        data = self.load_data()
        linear_user = tf.keras.models.load_model(
            f"/tmp/{self.hparams.model_dir}/linear_user"
        )
        linear_item = tf.keras.models.load_model(
            f"/tmp/{self.hparams.model_dir}/linear_item"
        )
        user_emb = tf.keras.models.load_model(f"/tmp/{self.hparams.model_dir}/user_emb")
        item_emb = tf.keras.models.load_model(f"/tmp/{self.hparams.model_dir}/item_emb")
        data = data.take(20).batch(20)

        fm_layer = FMLayer()

        # shape (batch_size, 1)
        user_linear_score = linear_user.predict(data)
        # shape (batch_size, user_field_size, embedding_size)
        user_vector = user_emb.predict(data)
        # reuse fm layer to calculate user interaction score
        # shape (batch_size, 1)
        user_vector_interaction_score = fm_layer(user_vector)
        # shape (batch_size, embedding_size)
        sum_user_vector = tf.math.reduce_sum(user_vector, axis=1)
        # shape (batch_size, 1)
        user_pad = tf.constant(1.0, shape=[tf.shape(sum_user_vector)[0], 1])
        # shape (batch_size, 1 + embedding_size)
        concat_user_vector = tf.concat([user_pad, sum_user_vector], axis=-1)

        # shape (batch_size, 1)
        item_linear_score = linear_item.predict(data)
        # shape (batch_size, item_field_size, 20)
        item_vector = item_emb.predict(data)
        # reuse fm layer to calculate item interaction score
        # shape (batch_size, 1)
        item_vector_interaction_score = fm_layer(item_vector)
        # shape (batch_size, embedding_size)
        sum_item_vector = tf.math.reduce_sum(item_vector, axis=1)
        # shape (batch_size, 1 + embedding_size)
        concat_item_vector = tf.concat(
            [item_linear_score + item_vector_interaction_score, sum_item_vector],
            axis=-1,
        )

        # calculate the inner product
        # shape (batch_size)
        scores = tf.reduce_sum(
            tf.multiply(concat_user_vector, concat_item_vector), axis=-1
        )
        # print(scores)

        # shape (batch_size)
        final_scores = tf.sigmoid(
            scores
            + tf.squeeze(user_linear_score)
            + tf.squeeze(user_vector_interaction_score)
        )

        main_model = tf.keras.models.load_model(f"/tmp/{self.hparams.model_dir}/main")
        # shape (batch_size)
        original_scores = tf.squeeze(main_model.predict(data))

        print(
            f"Do the two scores equal? The answer is: {tf.math.equal(final_scores, original_scores)}"
        )
