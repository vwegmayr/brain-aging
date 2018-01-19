"""
!!! WARNING !!!
When using in this script external values (config keys, files, ...)
make sure their content appears somehow in @get_data_preprocessing_values
function. So that when their value change, data is regenerated automatically.
"""
import numpy as np
import tensorflow as tf
import random


def get_data_preprocessing_values(config):
    """
    Returns a dictionnary with serializable values.
    Whenever the data should be re-generated, the content
    of the returned dictionnary should change.
    """
    import json
    import os
    return {
        'sources': [
            json.dumps(s.__dict__) for s in get_all_data_sources(config)
        ],
        'files': {
            os.path.basename(__file__): open(__file__).read(),
        },
        'config': config,
        'modules': {
            'tf': tf.__version__,
        }
    }


class DataAggregator:
    def __init__(self, config):
        self.config = config
        self.study_to_id = {}
        compression = getattr(
            tf.python_io.TFRecordCompressionType,
            config['dataset_compression'],
        )
        output_dir = config['data_converted_directory']
        self.writers = {
            'train': tf.python_io.TFRecordWriter(
                output_dir + config['train_database_file'],
                tf.python_io.TFRecordOptions(compression),
            ),
            'test': tf.python_io.TFRecordWriter(
                output_dir + config['test_database_file'],
                tf.python_io.TFRecordOptions(compression),
            ),
        }
        self.patient_to_writer = {}
        self.curr_study_id = -1
        self.curr_study_name = ''
        self.count = 0
        self.stats = {}

    def begin_study(self, study_name):
        self.study_to_id[study_name] = len(self.study_to_id)
        self.curr_study_id = self.study_to_id[study_name]
        self.curr_study_name = study_name
        self.stats[study_name] = {
            'success': 0,
            'errors': []
        }
        self.count = 1
        self.patient_to_writer = {}

    def get_writer_for_image(self, patient_id):
        if patient_id not in self.patient_to_writer:
            if random.random() < self.config['test_set_size_ratio']:
                self.patient_to_writer[patient_id] = self.writers['test']
            else:
                self.patient_to_writer[patient_id] = self.writers['train']
        return self.patient_to_writer[patient_id]

    def add_image(self, image_path, int64_features):
        import nibabel as nib
        import features
        Features = tf.train.Features
        Feature = tf.train.Feature
        Int64List = tf.train.Int64List
        Example = tf.train.Example

        if self.count % 10 == 1:
            print('[%s] Processing image #%d...' % (
                self.curr_study_name, self.count))
        self.count += 1

        img = nib.load(image_path)
        img_data = img.get_data()
        if list(img_data.shape) != list(self.config['image_shape']):
            self.add_error(
                image_path,
                'Image has shape %s, expected %s' % (
                    img_data.shape, self.config['image_shape'])
            )
            return

        # Transform features and write
        def _int64_to_feature(v):
            return Feature(int64_list=Int64List(value=[v]))
        img_features = {
            k: _int64_to_feature(v)
            for k, v in int64_features.items()
        }
        img_features[features.STUDY_ID] = _int64_to_feature(self.curr_study_id)
        img_features[features.MRI] = Feature(
            int64_list=Int64List(
                value=img_data.reshape([-1]).astype(np.int64)
            ),
        )

        # Check we have all features set
        assert(all(
            ft_name in img_features
            for ft_name in features.all_features.feature_info.keys()
        ))
        example = Example(features=Features(feature=img_features))
        self.get_writer_for_image(
            int64_features[features.STUDY_PATIENT_ID]
        ).write(example.SerializeToString())
        self.stats[self.curr_study_name]['success'] += 1

    def add_error(self, path, message):
        self.stats[self.curr_study_name]['errors'].append(message)
        print '[ERROR] %s (%s)' % (message, path)

    def finish(self):
        for w in self.writers.values():
            w.close()
        print '---- DATA CONVERSION STATS ----'
        for k, v in self.stats.items():
            print '%s: %d ok / %d errors' % (k, v['success'], len(v['errors']))
            if len(v['errors']) > 0:
                print '    First error:'
                print '    %s' % v['errors'][0]


class DataSource(object):
    def __init__(self, config):
        import glob
        self.config = config
        self.load_patients_features(config['patients_features'])
        self.all_files = glob.glob(config['glob'])

    def load_patients_features(self, csv_file_path):
        """
        Load Patients features from csv
        This file should contain the following columns:
        - id
        - One column per feature from features.py file, excluding the ones
          extracted from the file name (typically study id and image id)
        """
        import features
        import csv

        self.patients_ft = {}
        with open(csv_file_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.patients_ft[int(row['id'])] = {
                    features.AGE: int(row['age']),
                    features.SEX: int(row['sex']),
                }

    def preprocess(self, dataset):
        import re
        import features
        dataset.begin_study(self.config['name'])

        # MRI scans
        features_from_filename = self.config['features_from_filename']
        features_in_regexp = features_from_filename['features_group']
        assert(all([
            n in features.all_features.feature_info
            for n in features_in_regexp
        ]))
        extract_from_path = re.compile(features_from_filename['regexp'])
        for f in self.all_files:
            ft = {}
            # Add features from filename
            match = extract_from_path.match(f)
            if match is None:
                dataset.add_error(f, 'Regexp doesnt match')
                continue
            for ft_name, ft_group in features_in_regexp.items():
                ft[ft_name] = int(match.group(ft_group))
            if features.STUDY_PATIENT_ID not in ft:
                dataset.add_error(f, 'Regexp should provide patient id')
                continue
            # Add features from CSV
            patient_id = ft[features.STUDY_PATIENT_ID]
            if patient_id not in self.patients_ft:
                dataset.add_error(
                    f,
                    'No features for patient %d' % (patient_id),
                )
                continue
            ft.update(self.patients_ft[patient_id])
            dataset.add_image(f, ft)


def get_all_data_sources(config):
    return [
        DataSource(source_config)
        for source_config in config['data_sources']
    ]


def preprocess_all(config):
    print('[INFO] Extracting/preprocessing data...')
    random.seed(config['test_set_random_seed'])
    dataset = DataAggregator(config)
    data_sources = get_all_data_sources(config)
    for e in data_sources:
        e.preprocess(dataset)
    dataset.finish()


def preprocess_all_if_needed(config):
    import pickle
    pkl_file = config['data_converted_directory'] + "extractor_values.pkl"
    current_extractor_values = get_data_preprocessing_values(config)
    try:
        extracted_data_values = pickle.load(open(pkl_file, "rb"))
    except IOError:
        extracted_data_values = {}

    if extracted_data_values == current_extractor_values:
        print('[INFO] Extracted data is up-to-date. Skipping preprocessing :)')
        return
    print('[INFO] Extracted data is outdated.')
    preprocess_all(config)
    pickle.dump(current_extractor_values, open(pkl_file, "wb"))
