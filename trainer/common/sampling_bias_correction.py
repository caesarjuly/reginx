import tensorflow as tf


class SamplingBiasCorrection(tf.keras.layers.Layer):
    """A naive implementation of SamplingBiasCorrection.
    It supports the basic step estimation operation and returns the sampling probability for each key.
    Notice it doesn't support key expiration yet, so the memory will continue to grow if used in sequential training.
    """
    def __init__(self, lr=0.05, **kwargs):
        """Use one table to record the lastest step, aka the A table in paper
        Use another table to record the estimated step gap for each key, the B table in paper
        """
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
