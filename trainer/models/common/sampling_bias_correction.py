import tensorflow as tf


class SamplingBiasCorrection(tf.keras.layers.Layer):
    """A enhanced implementation of SamplingBiasCorrection.
    It supports the step estimation operation and returns the sampling probability for each key.
    The keys will be cleaned automatically while training when the vocabulary size reach the upper limit.
    """

    def __init__(self, lr=0.05, capacity=2**17, unload_factor=1.25, **kwargs):
        """Use one table to record the lastest step, aka the A table in paper
        Use another table to record the estimated step gap for each key, the B table in paper

        Args:
            lr (float, optional): learning rate. Defaults to 0.05.
            capacity (int, optional): required capacity. Defaults to 2**17.
            unload_factor (float, optional): unload factor, used to decide the upper threshold for triggering expiration. Defaults to 1.25.
        """
        super(SamplingBiasCorrection, self).__init__(**kwargs)
        self.lr = lr
        self.capacity = capacity
        self.unload_factor = unload_factor
        # calculate the capacity threshold
        self.upper_threshold_capacity = int(self.capacity * self.unload_factor)
        self.lastest_step = tf.lookup.experimental.MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=0,
        )
        self.step_gap = tf.lookup.experimental.MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.float32,
            default_value=0,
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
        if self.lastest_step.size() > self.upper_threshold_capacity:
            self.clean()
        return 1 / cur_gap

    def clean(self):
        """Clean the expired keys, shrink the table size to capacity"""
        ids, steps = self.lastest_step.export()
        ids = tf.reshape(ids, [-1])
        steps = tf.reshape(steps, [-1])
        cur_size = self.lastest_step.size()
        if cur_size > self.capacity:
            # find the top N ids with smallest ids, here we use the negative operator to reverse the search
            _, expired_indices = tf.math.top_k(
                tf.negative(steps), tf.cast(cur_size - self.capacity, tf.int32)
            )
            expired_ids = tf.gather(ids, expired_indices)
            self.lastest_step.remove(expired_ids)
            self.step_gap.remove(expired_ids)

    def size(self):
        return self.lastest_step.size()

    def export(self):
        return self.lastest_step.export()
