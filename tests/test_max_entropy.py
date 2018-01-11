import tensorflow as tf
from modules.models.trackers import MaxEntropyTracker


class TestMaxEntropy(tf.test.TestCase):
    """Test the maximum entropy implementation

    Focus on the loss.
    """

    def test_max_entropy_loss(self):
        with self.test_session():
            mu = [[1.0, 0]]
            y = [[2.0, 0]]
            T = float(1)
            k = float(2)
            loss = MaxEntropyTracker.max_entropy_loss(y, mu, k, T)
            print(loss.eval())

if __name__ == "__main__":
    tf.test.main()