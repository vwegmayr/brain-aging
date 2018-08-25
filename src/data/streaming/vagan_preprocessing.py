import numpy as np
import yaml
import os
from collections import OrderedDict
import tensorflow as tf

from .mri_streaming import MRISingleStream
from src.baum_vagan.vagan.model_wrapper import VAGanWrapper
from . import features as _features


def load_wrapper(smt_label):
    config_path = os.path.join("data", smt_label, "config.yaml")
    model_dir = os.path.join("data", smt_label, "logdir")
    with open(config_path, 'r') as f:
        model_config = yaml.load(f)
    wrapper = VAGanWrapper(**model_config)
    wrapper.vagan.load_weights(model_dir)

    return wrapper


class VaganFarPredictions(MRISingleStream):
    def __init__(self, *args, **kwargs):
        super(VaganFarPredictions, self).__init__(
            *args,
            **kwargs
        )

        # Load vagan
        smt_label = self.config["vagan_label"]
        self.wrapper = load_wrapper(smt_label)
        self.cached_computations = {}

    def cache_preprocessing(self):
        return self.config["cache_preprocessing"]

    def get_target_age(self):
        return self.config["target_age"]

    def preprocess_image(self, fid, im):
        target_age = self.get_target_age()
        cur_age = self.get_exact_age(fid)
        n_steps = int(target_age - cur_age)
        assert n_steps > 0
        images, masks = self.wrapper.vagan.iterated_far_prediction(im, n_steps)

        return np.squeeze(images[-1])

    def get_input_fn(self, mode):
        batches = self.get_batches(mode)
        groups = [group for batch in batches for group in batch]
        group_size = len(groups[0].file_ids)
        files = [group.file_ids for group in groups]

        # get feature names present in csv file (e.g. patient_label)
        # and added during preprocessing (e.g. file_name)
        feature_keys = self.file_id_to_meta[
            self.all_file_paths[0]
        ].keys()

        port_features = [
            k
            for k in feature_keys
            if (k != _features.MRI) and (k in self.feature_desc)
        ]

        def _read_files(file_ids, label):
            file_ids = [fid.decode('utf-8') for fid in file_ids]
            ret = []
            for fid in file_ids:
                path = self.get_file_path(fid)

                file_features = self.file_id_to_meta[fid]

                if (fid not in self.cached_computations) or \
                        not self.cache_preprocessing():
                    image = self.load_sample(path).astype(np.float32)
                    image = self.preprocess_image(fid, image)

                    if self.cache_preprocessing():
                        self.cached_computations[fid] = np.copy(image)
                else:
                    image = self.cached_computations[fid]

                ret += [image]

                ret += [
                    file_features[pf]
                    for pf in port_features
                ]
            # print("_read_files {}".format(ret[0] is None))
            return ret  # return list of features

        def _parser(*to_parse):
            if self.sample_shape is None:
                sample_shape = self.get_sample_shape()
            else:
                sample_shape = self.sample_shape
            el_n_features = 1 + len(port_features)  # sample + csv features
            all_features = OrderedDict()

            # parse features for every sample in group
            for i in range(group_size):
                self.feature_desc[_features.MRI]["shape"] = sample_shape
                mri_idx = i * el_n_features
                _mri = to_parse[mri_idx]
                ft = {
                    _features.MRI: tf.reshape(_mri, sample_shape),
                }

                ft.update({
                    port_features[i]: to_parse[mri_idx + i + 1]
                    for i in range(0, el_n_features - 1)
                })
                ft.update({
                    ft_name: d['default']
                    for ft_name, d in self.feature_desc.items()
                    if ft_name not in ft
                })
                el_features = {
                    ft_name + "_" + str(i): tf.reshape(
                        ft_tensor,
                        self.feature_desc[ft_name]['shape']
                    )
                    for ft_name, ft_tensor in ft.items()
                }  # return dictionary of features, should be tensors
                # rename mri_i to X_i
                el_features["X_" + str(i)] = el_features.pop(
                    _features.MRI + "_" + str(i)
                )
                all_features.update(el_features)

            return all_features

        labels = len(files) * [0]  # currently not used
        dataset = tf.data.Dataset.from_tensor_slices(
            tuple([files, labels])
        )

        # mri + other features
        read_types = group_size * ([tf.float32] + [
            self.feature_desc[fname]["type"]
            for fname in port_features
        ])

        num_calls = 4
        if "parallel_readers" in self.config:
            num_calls = self.config["parallel_readers"]
        dataset = dataset.map(
            lambda file_ids, label: tuple(tf.py_func(
                _read_files,
                [file_ids, label],
                read_types,
                stateful=False,
                name="read_files"
            )),
            num_parallel_calls=num_calls
        )

        prefetch = 4
        if "prefetch" in self.config:
            prefetch = self.config["prefetch"]
        dataset = dataset.map(_parser)
        dataset = dataset.prefetch(prefetch * self.config["batch_size"])
        dataset = dataset.batch(batch_size=self.config["batch_size"])

        def _input_fn():
            return dataset.make_one_shot_iterator().get_next()
        return _input_fn
