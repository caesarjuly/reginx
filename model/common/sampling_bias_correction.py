import tensorflow as tf


class SamplingBiasCorrection(tf.keras.layers.Layer):
    def __init__(self, lr=0.05, **kwargs):
        super(SamplingBiasCorrection, self).__init__(**kwargs)
        self.lr = lr
        self.lastest_step = tf.lookup.experimental.DenseHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=0,
            empty_key="",
            deleted_key="$",
        )
        self.step_gap = tf.lookup.experimental.DenseHashTable(
            key_dtype=tf.string,
            value_dtype=tf.float32,
            default_value=0,
            empty_key="",
            deleted_key="$",
        )

    def call(self, cur_step, candidate_ids):
        cur_step = tf.repeat(cur_step, tf.shape(candidate_ids))
        latest_step = self.lastest_step.lookup(candidate_ids)
        previous_gap = self.step_gap.lookup(candidate_ids)
        # if it's the first time meet this sample, then turn the lr to 1.0
        cur_gap = (1 - self.lr) * previous_gap + tf.where(
            latest_step == 0, 1.0, self.lr
        ) * tf.cast(cur_step - latest_step, tf.float32)
        self.lastest_step.insert(candidate_ids, cur_step)
        self.step_gap.insert(candidate_ids, cur_gap)
        return 1 / cur_gap
