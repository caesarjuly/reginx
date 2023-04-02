import unittest
import tensorflow as tf

from trainer.common.sampling_bias_correction import SamplingBiasCorrection


class TestSum(unittest.TestCase):
    def setUp(self):
        self.sbc = SamplingBiasCorrection(lr=0.5)
        # expiration logic test
        self.sbc1 = SamplingBiasCorrection(lr=0.5, capacity=10, unload_factor=1.25)

    def test_frequency_estimation(self):
        keys = tf.strings.as_string(tf.range(10))
        prob = self.sbc(cur_step=tf.constant(1, dtype=tf.int64), candidate_ids=keys)
        self.assertEqual([1.0] * 10, prob.numpy().tolist())
        # test the fist 5 items
        keys = tf.strings.as_string(tf.range(5))
        prob = self.sbc(cur_step=tf.constant(2, dtype=tf.int64), candidate_ids=keys)
        self.assertEqual([1.0] * 5, prob.numpy().tolist())
        # test the last 5 items
        keys = tf.strings.as_string(tf.range(5, 10))
        prob = self.sbc(cur_step=tf.constant(3, dtype=tf.int64), candidate_ids=keys)
        self.assertEqual(
            [0.667] * 5, list(map(lambda x: round(x, 3), prob.numpy().tolist()))
        )

    def test_clean(self):
        # put 10 keys for step 1
        keys = tf.strings.as_string(tf.range(10))
        self.sbc1(cur_step=tf.constant(1, dtype=tf.int64), candidate_ids=keys)
        # put another 5 keys for step 2
        keys = tf.strings.as_string(tf.range(10, 15))
        self.sbc1(cur_step=tf.constant(2, dtype=tf.int64), candidate_ids=keys)
        self.assertEqual(10, self.sbc1.size())
        # put another 5 keys for step 2
        keys = tf.strings.as_string(tf.range(15, 20))
        self.sbc1(cur_step=tf.constant(2, dtype=tf.int64), candidate_ids=keys)
        self.assertEqual(10, self.sbc1.size())
        # only the step 2 keys left
        keys, values = self.sbc1.export()
        self.assertSetEqual(
            set(map(str, range(10, 20))),
            set(map(lambda x: x.decode("utf-8"), keys.numpy())),
        )
        self.assertListEqual([2] * 10, list(values.numpy()))


if __name__ == "__main__":
    unittest.main()
