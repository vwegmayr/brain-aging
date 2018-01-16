import tensorflow as tf
import config


def parser(record):
    import features
    keys_to_features = {
        name: tf.FixedLenFeature(shape=info['shape'], dtype=info['type'])
        for (name, info) in features.all_features.feature_info.items()}

    parsed = tf.parse_single_example(record, features=keys_to_features)

    def process_feature(ft, ft_info):
        if ft_info['shape'] == []:
            return ft
        return tf.reshape(ft, ft_info['shape'])

    return {
        name: process_feature(parsed[name], info)
        for (name, info) in features.all_features.feature_info.items()
    }


def train_input():
    dataset = tf.data.TFRecordDataset(
        [config.prefix_data_converted + config.train_database_file],
        compression_type='GZIP',
    )
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(8)
    dataset = dataset.repeat(5)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def test_input():
    dataset = tf.data.TFRecordDataset(
        [config.prefix_data_converted + config.test_database_file],
        compression_type='GZIP',
    )
    dataset = dataset.map(parser)
    dataset = dataset.batch(8)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
