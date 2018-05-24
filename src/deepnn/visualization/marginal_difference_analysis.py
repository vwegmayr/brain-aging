import tensorflow as tf
import numpy as np
import nibabel as nib


class MarginalDifferenceAnalysis:
    def __init__(
        self,
        session,
        images_dataset,
        cnn_probas_output,
        cnn_feed_input,
        cnn_feed_other_values,
        small_hw=10,
        big_hw=15,
    ):
        """
        hf = Half-windows
        """
        BATCH_SIZE = 16
        self.session = session
        self.images_dataset = self.load_dataset(images_dataset)
        self.image_shape = self.images_dataset.get_shape().as_list()[1:]
        self.small_hw = small_hw
        self.big_hw = big_hw
        # Tensors from already existing graph
        self.cnn_probas_output = cnn_probas_output
        self.cnn_feed_input = cnn_feed_input
        self.cnn_feed_other_values = cnn_feed_other_values
        # TF Placeholders
        self.center_pos_placeholder = tf.placeholder(tf.float32, [3])
        self.analyzed_image_placeholder = tf.placeholder(
            tf.float32, self.image_shape)
        # TF Graph
        self.block_small_hw = self.create_block_mask(self.small_hw)
        self.block_big_hw = self.create_block_mask(self.big_hw)
        self.images_batch_tf = self.create_images_with_block_sampled(
            BATCH_SIZE)

    def load_dataset(self, file_names):
        sample_image = nib.load(file_names[0]).get_data()
        tf_dataset = tf.Variable(
            tf.zeros([len(file_names)] + list(sample_image.shape)),
            name="da_dataset",
            dtype=tf.float32,
            expected_shape=[len(file_names)] + list(sample_image.shape),
        )
        tf_i = tf.placeholder(dtype=tf.int32, shape=[])
        tf_image = tf.placeholder(dtype=tf.float32, shape=sample_image.shape)
        with tf.control_dependencies([tf.scatter_update(
            tf_dataset,
            tf_i,
            self._normalize_image(tf_image))
        ]):
            op = tf.identity(tf_i)
        print('Loading %d input images...' % len(file_names))
        self.session.run(tf.variables_initializer([tf_dataset]))
        for i, image_file_name in enumerate(file_names):
            self.session.run([op], {
                tf_i: i,
                tf_image: nib.load(image_file_name).get_data(),
            })
        print('Dataset loaded in GPU memory!')
        return tf_dataset

    def create_block_mask(self, block_hw):
        coords_x, coords_y, coords_z = np.mgrid[
            0:self.image_shape[0],
            0:self.image_shape[1],
            0:self.image_shape[2],
        ]
        coords_x = tf.convert_to_tensor(coords_x, dtype=tf.float32)
        coords_y = tf.convert_to_tensor(coords_y, dtype=tf.float32)
        coords_z = tf.convert_to_tensor(coords_z, dtype=tf.float32)
        center_x = tf.cast(self.center_pos_placeholder[0], tf.float32)
        center_y = tf.cast(self.center_pos_placeholder[1], tf.float32)
        center_z = tf.cast(self.center_pos_placeholder[2], tf.float32)
        in_range = [
            tf.logical_and(
                coords_x <= center_x + block_hw,
                center_x - block_hw <= coords_x,
            ),
            tf.logical_and(
                coords_y <= center_y + block_hw,
                center_y - block_hw <= coords_y,
            ),
            tf.logical_and(
                coords_z <= center_z + block_hw,
                center_z - block_hw <= coords_z,
            )
        ]
        return tf.logical_and(
            tf.logical_and(in_range[0], in_range[1]),
            in_range[2],
        )

    def create_images_with_block_sampled(self, batch_size):
        images_to_sample_from = tf.random_uniform(
            [batch_size],
            minval=0,
            maxval=self.images_dataset.get_shape().as_list()[0],
            dtype=tf.int32,
        )
        block_mask = tf.cast(
            tf.expand_dims(self.block_small_hw, axis=0),
            tf.float32,
        )
        result = tf.expand_dims(
            self._normalize_image(self.analyzed_image_placeholder),
            axis=0,
        ) * (1.0 - block_mask)
        result += tf.gather(
            self.images_dataset,
            images_to_sample_from,
        ) * block_mask
        return result

    def run_block(self, image, center_pos):
        images_batch, block_mask = self.session.run([
            self.images_batch_tf,
            self.block_small_hw,
        ], {
            self.center_pos_placeholder: center_pos,
            self.analyzed_image_placeholder: image,
        })
        feed_dict = {
            self.cnn_feed_input: images_batch,
        }
        feed_dict.update(self.cnn_feed_other_values)
        probas = self.session.run(self.cnn_probas_output, feed_dict)
        return probas, block_mask

    def compute_odds(self, raw_proba):
        NUM_CLASSES_K = 2
        NUM_TRAINING_SAMPLES_N = 300
        # Laplace correction
        raw_proba = (raw_proba * NUM_TRAINING_SAMPLES_N + 1)
        raw_proba /= float(NUM_TRAINING_SAMPLES_N + NUM_CLASSES_K)
        # Odds func
        return raw_proba / (1. - raw_proba)

    def visualize_image(self, image, const_z=None, step=None, class_index=0):
        assert(list(image.shape) == list(self.image_shape))
        if step is None:
            step = self.small_hw / 2
        we = np.zeros_like(image)
        counts = np.zeros_like(image)

        # Iterate all possible blocks
        margin = self.small_hw + 1
        all_centers_todo = []
        for center_x in range(margin, image.shape[0] - margin, step):
            for center_y in range(margin, image.shape[1] - margin, step):
                if const_z is not None:
                    all_centers_todo.append(
                        np.array([center_x, center_y, const_z]))
                    continue
                for center_z in range(margin, image.shape[2] - margin, step):
                    all_centers_todo.append(
                        np.array([center_x, center_y, center_z]))
        print('There are %d blocks to iterate over' % len(all_centers_todo))

        # Compute class probability with full image
        feed_dict = {
            self.cnn_feed_input: image.reshape([-1] + list(image.shape)),
        }
        feed_dict.update(self.cnn_feed_other_values)
        full_img_proba = self.session.run(self.cnn_probas_output, feed_dict)
        full_img_odds = self.compute_odds(full_img_proba[0, class_index])

        # Compute class probability when replacing part of the image
        for i, center_pos in enumerate(all_centers_todo):
            if i % 50 == 0:
                print('Doing block %d/%d...' % (i, len(all_centers_todo)))
            probas, block_mask = self.run_block(image, center_pos)
            proba = np.mean(probas[:, class_index])
            odd = self.compute_odds(proba)
            we += (
                np.log(full_img_odds) - np.log(odd)
            ) * block_mask.astype(np.float32)
            counts += block_mask

        print('Done: Visualization computed :)')
        return we / (counts.astype(np.float32) + 0.001)

    def _normalize_image(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
        return (x - mean) / tf.sqrt(var + 0.0001)
