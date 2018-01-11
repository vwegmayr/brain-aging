import tensorflow as tf
from modules.models.trackers import MaxEntropyTracker


class TestMaxEntropy(tf.test.TestCase):
    """Test the maximum entropy implementation

    Focus on the loss.
    """

    def test_max_entropy_loss(self):
        """Test computations done with Wolfram Mathematica."""
        with self.test_session():
            mu = [[1.0, 0]]
            y = [[2.0, 0]]
            T = float(1)
            k = float(1)
            loss = MaxEntropyTracker.max_entropy_loss(y, mu, k, T)
            self.assertAlmostEqual(loss.eval(), -3.005498894039818)

if __name__ == "__main__":
    tf.test.main()