"""
This module contains functions to handle the input data to the CNN model.
"""

import random
import pickle
import os
import re
import csv
import copy
import tensorflow as tf
import numpy as np
import nibabel as nb
import scipy.ndimage.interpolation as sni
import src.features as ft_def
from src.data.features_store import str_to_ft


class DataProvider(object):
    def __init__(self, input_fn_config=None):
        if input_fn_config is None:
            input_fn_config = {'py_streaming': {'classes': ['healthy', 'health_ad'], 'seed': 0, 'batch_size': 12, 'data_paths': {'regex': '_normalized\\.nii\\.gz', 'split_on': '_normalized.nii.gz', 'class_labels': '/local/ADNI_AIBL/ADNI_AIBL_T1_normalized/py2/AIBL_ADNI_class_labels_T1_NC_AD.pkl', 'train_data': '/local/ADNI_AIBL/ADNI_AIBL_T1_normalized/py2/AIBL_ADNI_train_T1_NC_AD.pkl', 'datadir': '/local/ADNI_AIBL/ADNI_AIBL_T1_normalized/train_NC_AD/', 'valid_data': '/local/ADNI_AIBL/ADNI_AIBL_T1_normalized/py2/AIBL_ADNI_valid_T1_NC_AD.pkl'}}, 'data_provider': 'py_streaming', 'image_shape': [91, 109, 91], 'data_generation': {'data_converted_directory': 'data/ready/', 'data_sources': [{'glob': '/local/ADNI_AIBL/ADNI_AIBL_T1_normalized/train/[0-9]*[0-9]_normalized*', 'name': 'ADNI_AIBL', 'features_from_filename': {'regexp': '.*/(\\d+)_normalized\\.nii\\.gz', 'features_group': {'study_image_id': 1}}, 'patients_features': 'data/raw/csv/adni_aibl__ad_hc.csv'}], 'image_normalization': {'outlier_percentile': 99, 'enable': True}, 'train_database_file': 'train.tfrecord', 'test_set_size_ratio': 0.2, 'test_set_random_seed': 0, 'dataset_compression': 'GZIP', 'test_database_file': 'test.tfrecord', 'train_test_split_on_feature': 'study_patient_id'}, 'data_streaming': {'dataset': [{'buffer_size': 400, 'call': 'prefetch'}, {'buffer_size': 500, 'call': 'shuffle'}, {'map_func': 'f', 'call': 'map', 'num_parallel_calls': 8}, {'map_func': 'f', 'call': 'map', 'num_parallel_calls': 8}, {'call': 'batch', 'batch_size': 8}]}}
        self.input_fn_config = input_fn_config
        self.inputs = {True: None, False: None}

        self.config = config = input_fn_config['py_streaming']
        self.random = random.Random()
        self.random.seed(config['seed'])
        train_files, test_files, self.file_to_features = get_train_test_filenames(config, self.random)
        test_files = [t for t in test_files if len(t) > 0]
        for i in range(len(train_files)):
            print("Train Class %d: %d samples (%s)" % (i, len(train_files[i]), config['classes'][i]))
        for i in range(len(test_files)):
            print("Valid Class %d: %d samples (%s)" % (i, len(test_files[i]), config['classes'][i]))

        for train, files in [(True, train_files), (False, test_files)]:
            self.inputs[train] = DataInput(
                config,
                train_files if train else test_files,
            )
            self.inputs[train].shuffle(self.random)
        self.mri_shape = nb.load(train_files[0][0]).get_data().shape

    def get_shuffled_filenames(self, train, shard):
        dataset = self.inputs[train].create_shard(shard)
        dataset.shuffle(self.random)
        all_files = []
        all_labels = []
        while not dataset.all_classes_looped():
            files, labels = dataset.next_batch_filenames(self.random)
            all_files += files
            all_labels += labels
        all_seeds = [
            self.random.randint(0, 1000000)
            for _ in all_labels
        ]
        return (all_files, all_labels, all_seeds)

    def get_input_fn(self, train, shard):
        classes = self.config['classes']
        ft_info = ft_def.all_features.feature_info
        port_features = [
            ft_def.IMAGE_LABEL,
            ft_def.SUBJECT_LABEL,
            ft_def.STUDY_IMAGE_ID,
            ft_def.STUDY_PATIENT_ID,
        ]

        def _read_files(f, label, seed):
            ft = self.file_to_features[f]
            ret = [DataInput.load_and_augment_file(f, seed)]
            ret += [
                ft[pf]
                for pf in port_features
            ]
            ret += [
                int(label == i)
                for i in range(len(classes))
            ]
            return ret

        def _parser(
            _mri,
            *other_values
        ):
            ported_features = other_values[:len(port_features)]
            classes_values = other_values[len(port_features):]
            ft = {
                ft_def.MRI: tf.reshape(_mri, self.mri_shape),
            }
            ft.update({
                port_features[i]: ported_features[i]
                for i in range(len(ported_features))
            })
            ft.update({
                c_ft: classes_values[c_id]
                for c_id, c_ft in enumerate(classes)
            })
            ft.update({
                ft_name: d['default']
                for ft_name, d in ft_def.all_features.feature_info.items()
                if ft_name not in ft
            })
            return {
                ft_name: tf.reshape(ft_tensor, ft_info[ft_name]['shape'])
                for ft_name, ft_tensor in ft.items()
            }

        dataset = tf.data.Dataset.from_tensor_slices(
            tuple(self.get_shuffled_filenames(train, shard))
        )
        dataset = dataset.map(
            lambda filename, label, seed: tuple(tf.py_func(
                _read_files,
                [filename, label, seed],
                [tf.float16] + [
                    ft_info[fname]['type']
                    for i, fname in enumerate(port_features)
                ] +
                [tf.int64] * len(classes),
                stateful=False,
                name='read_files',
            )),
            num_parallel_calls=12,
        )
        dataset = dataset.map(_parser)
        dataset = dataset.prefetch(10 * self.config['batch_size'])
        dataset = dataset.batch(batch_size=self.config['batch_size'])

        def _input_fn():
            return dataset.make_one_shot_iterator().get_next()
        return _input_fn

    def predict_features(self, features):
        return features

    def get_mri_shape(self):
        return list(self.mri_shape)


def get_train_test_filenames(config, r):
    paths = config['data_paths']
    # All patients class labels dictionary and list of validation patient codes
    patients_dict = pickle.load(open(paths['class_labels'], 'rb'))
    valid_patients = pickle.load(open(paths['valid_data'], 'rb'))
    train_patients = pickle.load(open(paths['train_data'], 'rb'))
    print("Validation patients count in Dict: ", len(valid_patients),
          "Train patients count in Dict:", len(train_patients))

    classes = config['classes']
    train_filenames = [[] for i in range(len(classes))]
    valid_filenames = [[] for i in range(len(classes))]

    for directory in os.walk(paths['datadir']):
        # Walk inside the directory
        for file in directory[2]:
            # Match all files ending with 'regex'
            input_file = os.path.join(directory[0], file)
            regex = r""+paths['regex']+"$"
            if re.search(regex, input_file):
                pat_code = input_file.rsplit(paths['split_on'])
                patient_code = pat_code[0].rsplit('/', 1)[1]
                if patient_code in train_patients:
                    train_filenames[patients_dict[patient_code]].append(
                        input_file)
                elif patient_code in valid_patients:
                    valid_filenames[patients_dict[patient_code]].append(
                        input_file)
                # else:
                #    print('NOTICE: Patient code %s not found' % patient_code)


    file_to_features = retrieve_features(
        [l2 for l1 in train_filenames for l2 in l1] +
        [l2 for l1 in valid_filenames for l2 in l1],
        **config['retrieve_features']
    )

    for i in range(len(classes)):
        if 'modify_train_set' in config:
            train_filenames[i] = modify_train_set(
                train_filenames[i],
                config['classes'][i],
                file_to_features,
                **config['modify_train_set']
            )
        if 'wrong_split' in config:
            c = config['wrong_split']
            _aug = c['skip_images_containing']
            groupped_by_image = {
                f: [f]
                for f in train_filenames[i]
                if _aug not in f
            }
            for f in train_filenames[i]:
                if _aug in f:
                    groupped_by_image[f.replace(_aug, '')].append(f)
            take_count = int(len(groupped_by_image) * c['test_ratio'])
            fnames_not_augmented = groupped_by_image.keys()
            r.shuffle(fnames_not_augmented)
            valid_filenames[i] = [img_f for img_f in fnames_not_augmented[:take_count]][:400]
            train_filenames[i] = [img_f for img_group in fnames_not_augmented[take_count:] for img_f in groupped_by_image[img_group]]
            #import ipdb; ipdb.set_trace()
        train_filenames[i].sort()
        valid_filenames[i].sort()
    return train_filenames, valid_filenames, file_to_features


class DataInput:
    """
    This class provides helper functions to manage the input datasets.
    Initialize this class with the required parameter file and the dataset
    as a tuple of filenames for each class.
    """
    def __init__(self, config, data, mean=0, var=0):
        self.config = config
        self.batch_size = config['batch_size']
        self.num_classes = len(data)
        self.files = [data[i] for i in range(0, self.num_classes)]
        self.batch_index = [0 for i in range(0, self.num_classes)]
        self.batch_looped = [0 for i in range(0, self.num_classes)]
        self.mean = mean
        self.var = var

    def all_classes_looped(self):
        return all([l > 0 for l in self.batch_looped])

    def shuffle(self, r):
        for class_label in range(0, self.num_classes):
            shuffle_indices = list(range(len(self.files[class_label])))
            r.shuffle(shuffle_indices)
            self.files[class_label] = [
                self.files[class_label][i]
                for i in shuffle_indices
            ]

    def create_shard(self, shard):
        child = DataInput(self.config, [[] for i in range(self.num_classes)])
        # Shard
        if shard is not None:
            index = shard[0]
            total = shard[1]
            for c in range(self.num_classes):
                child.files[c] = [
                    f
                    for i, f in enumerate(self.files[c])
                    if (i % total) == index
                ]
        else:
            child.files = copy.copy(self.files)
        return child

    def next_batch_filenames(self, r):
        """
        This functions retrieves the next batch of the data.

        Returns: (batch_filenames, batch_labels)

        """
        batch_files = []
        batch_labels = []

        # Increment batch_size/num_class for each class
        inc = int(self.batch_size / self.num_classes)
        inc = 1
        # If unbalanced, then increment the images of a specific class, say 0
        unbalanced = self.batch_size % self.num_classes
        batch_order = []
        for i in range(0, self.num_classes):
            batch_order += [
                i
                for j in range(0, int(self.batch_size / self.num_classes))
            ]
        r.shuffle(batch_order)
        for class_label in batch_order:
            start = self.batch_index[class_label]
            class_files = []
            if unbalanced != 0 and class_label == 0:
                end = start + inc + 1
                self.batch_index[class_label] += inc + 1
            else:
                end = start + inc
                self.batch_index[class_label] += inc

            if end > len(self.files[class_label]):
                # Reached end of epoch
                class_files = [
                    self.files[class_label][i]
                    for i in range(start, len(self.files[class_label]))
                ]
                left_files = end - len(self.files[class_label])
                shuffle_indices = list(range(len(self.files[class_label])))
                r.shuffle(shuffle_indices)
                self.files[class_label] = [
                    self.files[class_label][i]
                    for i in shuffle_indices
                ]
                start = 0
                end = left_files
                self.batch_index[class_label] = left_files
                self.batch_looped[class_label] += 1
            for i in range(start, end):
                class_files.append(self.files[class_label][i])
            batch_files += class_files
            batch_labels += [class_label] * (end - start)
        assert(len(batch_files) == self.batch_size)
        assert(len(batch_labels) == self.batch_size)
        return batch_files, batch_labels

    @staticmethod
    def rotate(mri_image, r):
        angle_rot = r.uniform(-3, 3)
        direction = r.choice(['x', 'y', 'z'])
        if direction == 'x':
            return sni.rotate(mri_image, angle_rot, (0, 1), reshape=False)
        if direction == 'y':
            return sni.rotate(mri_image, angle_rot, (0, 2), reshape=False)
        if direction == 'z':
            return sni.rotate(mri_image, angle_rot, (1, 2), reshape=False)

    @staticmethod
    def translate(mri_image, r):
        pixels = r.uniform(-4, 4)
        direction = r.choice(['x', 'y', 'z'])
        if direction == 'x':
            return sni.shift(mri_image, [pixels, 0, 0], mode='nearest')
        if direction == 'y':
            return sni.shift(mri_image, [0, pixels, 0], mode='nearest')
        if direction == 'z':
            return sni.shift(mri_image, [0, 0, pixels], mode='nearest')

    @staticmethod
    def load_and_augment_file(filename, seed, augment=False):
        mri_image = nb.load(filename)
        mri_image = mri_image.get_data()

        # Data augmentation
        if augment:
            r = random.Random()
            r.seed(seed)
            mri_image = DataInput.translate(mri_image, r)
            mri_image = DataInput.rotate(mri_image, r)
        return mri_image.astype(np.float16)


def retrieve_features(
    dataset,
    patients_csv,
    regex_extract_image_id,
):
    image_id_to_features = {}
    with open(patients_csv) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row = {
                k: str_to_ft(k, v)
                for k, v in row.items()
                if k in ft_def.all_features.feature_info
            }
            row.update({
                ft_name: d['default']
                for ft_name, d in ft_def.all_features.feature_info.items()
                if ft_name not in row
            })
            image_id_to_features[int(row[ft_def.STUDY_IMAGE_ID])] = row
    regex = re.compile(regex_extract_image_id)
    file_to_features = {}
    for file in dataset:
        image_id = int(regex.match(file).group(1))
        patient_id = image_id_to_features[image_id][ft_def.STUDY_PATIENT_ID]
        file_to_features[file] = image_id_to_features[image_id]
    return file_to_features


def map_patient_to_files(train_set, file_to_features):
    set_patient_to_images = {}
    for f in train_set:
        patient_id = file_to_features[f][ft_def.STUDY_PATIENT_ID]
        if patient_id not in set_patient_to_images:
            set_patient_to_images[patient_id] = []
        set_patient_to_images[patient_id].append(f)
    return set_patient_to_images


def modify_train_set(
    train_set,
    class_name,
    file_to_features,
    keep_patients=None,
    max_images_per_patient=None,
    min_images_per_patient=None,
    maximum_total_files=None,
    seed=0,
):
    r = random.Random(seed)
    # Group images by patient_id
    set_patient_to_images = map_patient_to_files(train_set, file_to_features)
    # For every patient, limit number of images
    if max_images_per_patient is not None:
        for patient_id in set_patient_to_images.keys():
            set_patient_to_images[patient_id].sort()
            r.shuffle(set_patient_to_images[patient_id])
            set_patient_to_images[patient_id] = \
                set_patient_to_images[patient_id][:max_images_per_patient]
    if min_images_per_patient is not None:
        set_patient_to_images = {
            k: v
            for k, v in set_patient_to_images.items()
            if len(v) >= min_images_per_patient
        }
    # Select patients
    take_patients = set_patient_to_images.keys()
    r.shuffle(take_patients)
    if keep_patients is not None:
        take_patients = take_patients[:keep_patients]
    total_files = 0
    for p in take_patients:
        total_files += len(set_patient_to_images[p])
    if maximum_total_files is None or maximum_total_files > total_files:
        maximum_total_files = total_files
    # Take all the files of selected patients
    max_reps = 0
    train_set = []
    while len(train_set) < maximum_total_files:
        # Add a file from every patient
        for patient_id in take_patients:
            if max_reps < len(set_patient_to_images[patient_id]):
                train_set.append(set_patient_to_images[patient_id][max_reps])
                if len(train_set) == maximum_total_files:
                    break
        max_reps += 1
    r.shuffle(train_set)
    # Debug print
    set_patient_to_images = map_patient_to_files(train_set, file_to_features)
    counts, number_patients = np.unique([
            len(v) for v in set_patient_to_images.values()
        ],
        return_counts=True,
    )
    print('  %s/train_set filtering' % class_name)
    for i in range(len(counts)):
        if i > 3:
            print('    ... up to %d samples each' % max_reps)
            break
        print('    %d patients with %d samples each' % (
            number_patients[i], counts[i],
        ))
    # Append some features
    return train_set
