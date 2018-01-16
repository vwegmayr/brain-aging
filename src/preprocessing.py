import numpy as np
import tensorflow as tf
import config
import random

random.seed(0)

class DataAggregator:
    def __init__(self, output_dir):
        self.study_to_id = {}
        self.writers = {
            'train': tf.python_io.TFRecordWriter(
                output_dir + config.train_database_file,
                tf.python_io.TFRecordOptions(config.dataset_compression),
            ),
            'test': tf.python_io.TFRecordWriter(
                output_dir + config.test_database_file,
                tf.python_io.TFRecordOptions(config.dataset_compression),
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
            if random.random() < config.test_set_size_ratio:
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
        if img_data.shape != config.image_shape:
            self.add_error(
                image_path,
                'Image has shape %s, expected %s' % (
                    img_data.shape, config.image_shape)
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


def preprocess_koln(dataset):
    import glob
    import re
    import csv
    import features
    import config
    dataset.begin_study('KOLN_T1')
    # Load Koln Patients features
    # id,age,sex,symptom_onset,disease_duration,updrs_3_on,updrs_3_off,
    #    led,ankk_1,drd_3,tar_score
    koln_patients_ft = {}
    with open(
        config.prefix_data_raw + 'KOLN_PATIENTS/patients.csv'
    ) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            koln_patients_ft[int(row['id'])] = {
                features.AGE: int(row['age']),
                features.SEX: int(row['sex']),
            }

    # MRI scans
    extract_patient_id = re.compile(r".*/(\d+)/(\d+)_t1\.nii\.gz")
    for f in glob.glob(config.prefix_data_raw + 'KOLN_T1/*/*/*.nii.gz'):
        match = extract_patient_id.match(f)
        patient_id = int(match.group(2))
        if patient_id not in koln_patients_ft:
            dataset.add_error(f, 'No features for patient %d' % (patient_id))
            continue
        ft = koln_patients_ft[patient_id]
        ft.update({
                features.STUDY_PATIENT_ID: int(match.group(2)),
                features.STUDY_IMAGE_ID: int(match.group(1)),
        })
        dataset.add_image(f, ft)


def preprocess_all():
    dataset = DataAggregator(config.prefix_data_converted)
    preprocess_koln(dataset)
    dataset.finish()
