import tensorflow as tf
import numpy as np
import nibabel as nib


class MarginalDifferenceAnalysis(object):
    def __init__(
        self,
        session,
        images_dataset,
        cnn_probas_output,
        cnn_feed_input,
        cnn_feed_other_values,
        step=4,
        small_hw=10,
        big_hw=15,
    ):
        """
        hf = Half-windows
        """
        BATCH_SIZE = 16
        self.session = session
        self.images_dataset = images_dataset
        self.image_shape = self.images_dataset.get_shape().as_list()[1:]
        self.small_hw = small_hw
        self.big_hw = big_hw
        self.step = step
        # Tensors from already existing graph
        self.cnn_probas_output = cnn_probas_output
        self.cnn_feed_input = cnn_feed_input
        self.cnn_feed_other_values = cnn_feed_other_values
        # TF Placeholders
        self.center_pos_placeholder = tf.placeholder(tf.float32, [3])
        self.hw_placeholder = tf.placeholder(tf.float32, [3])
        self.analyzed_image_placeholder = tf.placeholder(
            tf.float32, self.image_shape)
        # TF Graph
        self.block_mask = self.create_block_mask(self.hw_placeholder)
        self.images_batch_tf = self.create_images_with_block_sampled(
            BATCH_SIZE)

    @staticmethod
    def load_dataset(file_names, session):
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
            MarginalDifferenceAnalysis._normalize_image(tf_image))
        ]):
            op = tf.identity(tf_i)
        print('Loading %d input images...' % len(file_names))
        session.run(tf.variables_initializer([tf_dataset]))
        for i, image_file_name in enumerate(file_names):
            session.run([op], {
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
                coords_x < center_x + block_hw[0],
                center_x - block_hw[0] <= coords_x,
            ),
            tf.logical_and(
                coords_y < center_y + block_hw[1],
                center_y - block_hw[1] <= coords_y,
            ),
            tf.logical_and(
                coords_z < center_z + block_hw[2],
                center_z - block_hw[2] <= coords_z,
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
            tf.expand_dims(self.block_mask, axis=0),
            tf.float32,
        )
        result = tf.expand_dims(
            MarginalDifferenceAnalysis._normalize_image(
                self.analyzed_image_placeholder),
            axis=0,
        ) * (1.0 - block_mask)
        result += tf.gather(
            self.images_dataset,
            images_to_sample_from,
        ) * block_mask
        return result

    def run_block(self, image, block_info):
        feed_dict = {
            self.center_pos_placeholder: block_info['center'],
            self.analyzed_image_placeholder: image,
            self.hw_placeholder: block_info['hw'],
        }
        images_batch, block_mask = self.session.run([
            self.images_batch_tf,
            self.block_mask,
        ], feed_dict)
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

    def visualize_image(self, image, const_z=None, class_index=0):
        assert(list(image.shape) == list(self.image_shape))
        we = np.zeros_like(image)
        counts = np.zeros_like(image)

        # Iterate all possible blocks
        all_todo = self._generate_all_centers_and_hw(const_z)

        # Compute class probability with full image
        feed_dict = {
            self.cnn_feed_input: image.reshape([-1] + list(image.shape)),
        }
        feed_dict.update(self.cnn_feed_other_values)
        full_img_proba = self.session.run(self.cnn_probas_output, feed_dict)
        full_img_odds = self.compute_odds(full_img_proba[0, class_index])

        # Compute class probability when replacing part of the image
        for i, block_info in enumerate(all_todo):
            if i % 500 == 0:
                print('Doing block %d/%d...' % (i, len(all_todo)))
            probas, block_mask = self.run_block(image, block_info)
            proba = np.mean(probas[:, class_index])
            odd = self.compute_odds(proba)
            we += (
                np.log(full_img_odds) - np.log(odd)
            ) * block_mask.astype(np.float32)
            counts += block_mask

        return we / (counts.astype(np.float32) + 0.001)

    def _generate_all_centers_and_hw(self, const_z):
        all_centers_todo = []
        margin = self.small_hw + 1
        image_shape = self.image_shape
        step = self.step
        for center_x in range(margin, image_shape[0] - margin, step):
            for center_y in range(margin, image_shape[1] - margin, step):
                if const_z is not None:
                    all_centers_todo.append({
                        'center': np.array([center_x, center_y, const_z]),
                        'hw': np.array([self.small_hw] * 3),
                    })
                    continue
                for center_z in range(margin, image_shape[2] - margin, step):
                    all_centers_todo.append({
                        'center': np.array([center_x, center_y, center_z]),
                        'hw': np.array([self.small_hw] * 3),
                    })
        return all_centers_todo

    @staticmethod
    def _normalize_image(x):
        mean, var = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
        return (x - mean) / tf.sqrt(var + 0.0001)


def next_multiple(x, m):
    return (int((x - 1) / m) + 1) * m


class PyramidalMDA(MarginalDifferenceAnalysis):
    def __init__(
        self,
        min_depth=1,
        max_depth=4,
        overlap_count=1,
        *args,
        **kwargs
    ):
        super(PyramidalMDA, self).__init__(
            *args, **kwargs)
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.overlap_count = overlap_count

    @staticmethod
    def _binary_splits(min_depth, max_depth, overlap_count, max_dim):
        """
        Returns a list of blocks in the format:
        [[block1_center, block1_hw], [block2_center, block2_hw], ...]
        """
        all_splits = []
        for d in range(min_depth, max_depth+1):
            window_size = max_dim / pow(2, d)
            window_size = next_multiple(window_size, 2)
            for shift_idx in range(overlap_count):
                shift = int(window_size * shift_idx / float(overlap_count))
                window_begin_coord = - window_size + shift
                while window_begin_coord < max_dim:
                    all_splits.append([
                        window_begin_coord + window_size / 2,  # center
                        window_size / 2,  # hw
                    ])
                    window_begin_coord += window_size
        return all_splits

    def _generate_all_centers_and_hw(self, const_z):
        all_centers_todo = []
        splits = [PyramidalMDA._binary_splits(
                self.min_depth,
                self.max_depth,
                self.overlap_count,
                self.image_shape[dim],
            )
            for dim in range(3)
        ]
        for x in splits[0]:
            for y in splits[1]:
                for z in splits[2]:
                    if const_z is not None and (
                        const_z < z[0] - z[1] or
                        const_z > z[0] + z[1]
                    ):
                        continue
                    all_centers_todo.append({
                        'center': np.array([x[0], y[0], z[0]]),
                        'hw': np.array([x[1], y[1], z[1]]),
                    })
        return all_centers_todo
