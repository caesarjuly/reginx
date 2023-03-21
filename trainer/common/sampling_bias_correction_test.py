import unittest
import tensorflow as tf

from trainer.common.sampling_bias_correction import SamplingBiasCorrection


class TestSum(unittest.TestCase):
    def setUp(self):
        self.sbc = SamplingBiasCorrection(lr=0.5)

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


if __name__ == "__main__":
    unittest.main()
