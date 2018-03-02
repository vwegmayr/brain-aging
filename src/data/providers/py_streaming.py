"""
This module contains functions to handle the input data to the CNN model.
"""

import random
import pickle
import os
import re
import copy
import tensorflow as tf
import numpy as np
import nibabel as nb
import scipy.ndimage.interpolation as sni
import src.features as ft_def


class DataProvider(object):
    def __init__(self, input_fn_config):
        self.input_fn_config = input_fn_config
        self.inputs = {True: None, False: None}

        self.config = config = input_fn_config['py_streaming']
        random_state = random.getstate()
        random.seed(config['seed'])
        train_files, test_files = get_train_test_filenames(config)

        for train, files in [(True, train_files), (False, test_files)]:
            self.inputs[train] = DataInput(
                config,
                train_files if train else test_files,
            )
            self.inputs[train].shuffle()
        random.setstate(random_state)
        self.mri_shape = nb.load(train_files[0][0]).get_data().shape

    def get_input_fn(self, train, shard):
        # Generate batch filenames
        dataset = self.inputs[train].create_shard_shuffled(shard)
        all_files = []
        all_labels = []
        while not dataset.all_classes_looped():
            files, labels = dataset.next_batch_filenames()
            all_files += files
            all_labels += labels

        def _read_files(f, label):
            return [
                DataInput.load_and_augment_file(f),
                int(label == 0),
                int(label == 2),
            ]

        def _parser(_mri, _healthy, _health_ad):
            return {
                ft_def.MRI: tf.reshape(_mri, self.mri_shape),
                ft_def.HEALTHY: _healthy,
                ft_def.HEALTH_AD: _health_ad,
            }
        dataset = tf.data.Dataset.from_tensor_slices((all_files, all_labels))
        dataset = dataset.map(
            lambda filename, label: tuple(tf.py_func(
                _read_files,
                [filename, label],
                [tf.float16, tf.int64, tf.int64],
                name='read_files',
            ))
        )
        dataset = dataset.map(_parser)
        dataset = dataset.batch(batch_size=self.config['batch_size'])

        def _input_fn():
            return dataset.make_one_shot_iterator().get_next()
        return _input_fn

    def predict_features(self, features):
        return features
# End of interface functions


def get_train_test_filenames(config):
    paths = config['data_paths']
    # All patients class labels dictionary and list of validation patient codes
    patients_dict = pickle.load(open(paths['class_labels'], 'rb'))
    valid_patients = pickle.load(open(paths['valid_data'], 'rb'))
    train_patients = pickle.load(open(paths['train_data'], 'rb'))
    print("Validation patients count in Dict: ", len(valid_patients),
          "Train patients count in Dict:", len(train_patients))

    classes = config['classes']
    train_filenames = [[] for i in range(0, classes)]
    valid_filenames = [[] for i in range(0, classes)]

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
                if patient_code in valid_patients:
                    valid_filenames[patients_dict[patient_code]].append(
                        input_file)

    for i in range(0, classes):
        print("Train Class ", i, len(train_filenames[i]))
        print("Valid Class ", i, len(valid_filenames[i]))
    return train_filenames, valid_filenames


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

    def shuffle(self):
        for class_label in range(0, self.num_classes):
            shuffle_indices = list(range(len(self.files[class_label])))
            random.shuffle(shuffle_indices)
            self.files[class_label] = [
                self.files[class_label][i]
                for i in shuffle_indices
            ]

    def create_shard_shuffled(self, shard):
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
        child.shuffle()
        return child

    def next_batch_filenames(self):
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
        random.shuffle(batch_order)
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
                random.shuffle(shuffle_indices)
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
    def rotate(filename, direction):
        angle_rot = random.uniform(-3, 3)
        mri_image = nb.load(filename).get_data()
        if direction == 'x':
            return sni.rotate(mri_image, angle_rot, (0, 1), reshape=False)
        if direction == 'y':
            return sni.rotate(mri_image, angle_rot, (0, 2), reshape=False)
        if direction == 'z':
            return sni.rotate(mri_image, angle_rot, (1, 2), reshape=False)

    @staticmethod
    def translate(filename, direction):
        pixels = random.uniform(-4, 4)
        mri_image = nb.load(filename).get_data()
        if direction == 'x':
            return sni.shift(mri_image, [pixels, 0, 0], mode='nearest')
        if direction == 'y':
            return sni.shift(mri_image, [0, pixels, 0], mode='nearest')
        if direction == 'z':
            return sni.shift(mri_image, [0, 0, pixels], mode='nearest')

    @staticmethod
    def load_and_augment_file(filename):
        # For augmentation
        mri_image = []
        if 'rot' in filename:
            split_filename = filename.split('rot')
            mri_image = DataInput.rotate(
                split_filename[0],
                split_filename[1],
            )
        elif 'trans' in filename:
            split_filename = filename.split('trans')
            mri_image = DataInput.translate(
                split_filename[0],
                split_filename[1],
            )

        else:
            mri_image = nb.load(filename)
            mri_image = mri_image.get_data()
        return mri_image.astype(np.float16)
