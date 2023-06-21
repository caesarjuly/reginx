from trainer.common.gcp import BUCKET_NAME, download_from_directory
from trainer.tasks.base import BaseTask
import tensorflow as tf


class GenerateFMEmbedding(BaseTask):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)

    def load_data(self) -> tf.data.Dataset:
        download_from_directory(BUCKET_NAME, self.hparams.data, "/tmp/train")
        return tf.data.Dataset.load("/tmp/train")

    def run(self):
        data = self.load_data()
        linear_item = tf.keras.models.load_model(
            f"/tmp/{self.hparams.model_dir}/linear_item"
        )
        user_emb = tf.keras.models.load_model(f"/tmp/{self.hparams.model_dir}/user_emb")
        item_emb = tf.keras.models.load_model(f"/tmp/{self.hparams.model_dir}/item_emb")
        data = data.take(20).batch(20)

        user_vector = tf.math.reduce_sum(user_emb.predict(data), axis=1)
        user_pad = tf.constant(1.0, shape=[tf.shape(user_vector)[0], 1])
        output_user_vector = tf.concat([user_pad, user_vector], axis=-1)
        # print(output_user_vector[0])

        item_score = linear_item.predict(data)
        # print(item_score[0])
        item_vector = tf.math.reduce_sum(item_emb.predict(data), axis=1)
        # print(item_vector[0])
        output_item_vector = tf.concat([item_score, item_vector], axis=-1)
        # print(output_item_vector[0])
        scores = tf.reduce_sum(
            tf.multiply(output_user_vector, output_item_vector), axis=-1
        )

        print(scores)
        print(tf.shape(scores))
