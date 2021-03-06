import nibabel as nib
import warnings
import numpy as np
import itertools
from time import process_time
from collections import OrderedDict
import os
import copy
import pandas as pd

from .base import FileStream
from .base import Group
from src.baum_vagan.utils import map_image_to_intensity_range


def merge_list_of_lists_by_size(l1, l2):
    ll1 = sum([len(e) for e in l1])
    ll2 = sum([len(e) for e in l2])

    if ll2 == 0:
        return l1
    elif ll1 == 0:
        return l2

    if ll1 > ll2:
        tmp = l1
        l1 = l2
        l2 = tmp

    ll1 = sum([len(e) for e in l1])
    ll2 = sum([len(e) for e in l2])
    r = ll2 / ll1

    l1_p = 0
    l1_s = 0
    l2_p = 0
    l2_s = 0
    res = []
    while (l1_p < len(l1) and l2_p < len(l2)):
        if l2_s < r * l1_s:
            res.append(l2[l2_p])
            l2_s += len(l2[l2_p])
            l2_p += 1
        else:
            res.append(l1[l1_p])
            l1_s += len(l1[l1_p])
            l1_p += 1

    res.extend(l1[l1_p:])
    res.extend(l2[l2_p:])

    return res


class MRIImageLoader(object):
    def load_image(self, image_path):
        mri_image = nib.load(image_path)
        mri_image = mri_image.get_data()

        return mri_image.astype(np.float32)


class MRISingleStream(FileStream, MRIImageLoader):
    def __init__(self, *args, **kwargs):
        super(MRISingleStream, self).__init__(
            *args,
            **kwargs
        )
        # train-test split is performed by parent
        self.rescale_to_one = self.config["rescale_to_one"]
        self.normalization_computed = False
        self.normalize_images = self.config["normalize_images"]
        if self.normalize_images:
            self.compute_image_normalization()

    def group_data(self, train_ids, test_ids):
        # We just to stream the images one by one
        groups = []
        for fid in train_ids:
            g = Group([fid])
            g.is_train = True
            groups.append(g)

        for fid in test_ids:
            g = Group([fid])
            g.is_train = False
            groups.append(g)

        return groups

    def do_clip(self):
        return "clip" in self.config and self.config["clip"]

    def clip_image(self, img):
        return np.clip(img, np.min(img), np.percentile(img, 95))

    def compute_image_normalization(self):
        """
        1. Normalize every image to 0 mean and 1 std.
        2. Normalize every voxel accross the whole dataset.
        3. Normalize every image again.

        Normalization should be computed on the train set only.
        """
        # collect train file IDs
        test_ids = set(self.get_test_ids())
        validation_ids = set(self.get_validation_ids())
        # some files may not be used by sampler, but all available
        # images should be used to compute the normalization
        file_ids = set(self.all_file_ids).difference(test_ids)
        file_ids = file_ids.difference(validation_ids)
        n = len(file_ids)

        shape = self.get_sample_shape()
        voxel_mean = np.zeros(shape)
        voxel_mean_sq = np.zeros(shape)

        lim = int(0.1 * n)
        # Try to avoid overflows
        cur_sums = np.zeros(shape)
        cur_sums_sq = np.zeros(shape)
        c = 0
        file_ids = sorted(list(file_ids))
        for fid in file_ids:
            p = self.get_file_path(fid)
            im = self.load_sample(p)
            if self.do_clip():
                im = self.clip_image(im)
            std = np.std(im)

            if np.isclose(std, 0):
                std = 1
            im = (im - np.mean(im)) / std

            # Keep running average for every voxel intensity
            cur_sums += im
            cur_sums_sq += im ** 2
            c += 1
            if c >= lim:
                c = 0
                voxel_mean += cur_sums / n
                voxel_mean_sq += cur_sums_sq / n
                cur_sums = np.zeros(shape)
                cur_sums_sq = np.zeros(shape)

        if c != 0:
            voxel_mean += cur_sums / n
            voxel_mean_sq += cur_sums_sq / n

        self.voxel_means = np.zeros(shape)
        self.voxel_stds = np.zeros(shape)

        # Compute mean and standard deviation
        self.voxel_means = voxel_mean
        voxel_var = voxel_mean_sq - self.voxel_means ** 2
        self.voxel_stds = np.sqrt(voxel_var)

        if not self.silent:
            print(">>>>> Normalization computed!!")
            print(">> min std voxel {}".format(np.min(self.voxel_stds)))

        self.normalization_computed = True

    def get_voxel_means(self):
        return self.voxel_means

    def get_voxel_stds(self):
        return self.voxel_stds

    def normalize_image(self, im):
        assert self.normalization_computed
        if self.do_clip():
            im = self.clip_image(im)

        std = np.std(im)
        if np.isclose(0, std):
            std = 1
        im = (im - np.mean(im)) / std
        im = (im - self.voxel_means) / self.voxel_stds
        std = np.std(im)
        if np.isclose(0, std):
            std = 1
        im = (im - np.mean(im)) / std
        return im

    def dump_normalization(self, outdir):
        if self.normalization_computed and self.normalize_images:
            np.savez_compressed(
                os.path.join(outdir, "normalization"),
                mean=self.voxel_means,
                std=self.voxel_stds
            )

    def load_raw_sample(self, file_path):
        im = self.load_image(file_path)
        if self.config["downsample"]["enabled"]:
            # im = im / np.max(im)
            shape = tuple(self.config["downsample"]["shape"])
            # im = resize(im, shape, anti_aliasing=False)
            im = self.np_random.rand(*shape)

        return im

    def load_sample(self, file_path):
        im = self.load_image(file_path)
        if self.config["downsample"]["enabled"]:
            # im = im / np.max(im)
            shape = tuple(self.config["downsample"]["shape"])
            # im = resize(im, shape, anti_aliasing=False)
            im = np.random.rand(*shape)
            return im

        if self.rescale_to_one:
            im = map_image_to_intensity_range(im, -1, 1, 5)

        if self.load_only_slice():
            slice_axis, slice_idx = self.get_slice_info()
            im = np.take(im, slice_idx, axis=slice_axis)

        return im

    def load_normalized_sample(self, file_path):
        im = self.load_image(file_path)
        return self.normalize_image(im)

    def group_stats(self, group, categorical, numerical):
        stats = {
            "n": len(group.file_ids)
        }
        all_keys = categorical + numerical
        is_cat_bool = len(categorical) * [True] + len(numerical) * [False]

        for k, is_cat in zip(all_keys, is_cat_bool):
            stats[k] = {
                "count": 0,
                "vals": []
            }
            for fid in group.file_ids:
                val = self.get_meta_info_by_key(fid, k)
                if is_cat and val == 1:
                    stats[k]["count"] += 1
                if not is_cat:
                    stats[k]["count"] += val
                    stats[k]["vals"].append(val)
            stats[k]["mean"] = stats[k]["count"] / stats["n"]
            if not is_cat:
                stats[k]["std"] = np.std(stats[k]["vals"])

        return stats

    def make_balanced_k_fold_split(self, patient_groups):
        """
        Make k folds for which the label distribution of the labels
        is close to the one of the complete data set.
        """
        k = self.config["n_folds"]
        categorical = self.config["categorical_split"]
        numerical = self.config["numerical_split"]
        further_stats = ["distinct_patients"]
        all_labels = categorical + numerical
        all_label_mean = {}
        all_label_std = {}
        all_vals = {
            label: []
            for label in all_labels
        }

        # Group by patient
        all_groups = patient_groups
        all_file_ids = [fid for g in all_groups for fid in g.file_ids]
        self.np_random.shuffle(all_groups)

        # Collect groups stats
        n_total_patients = 0
        for group in all_groups:
            stats = group.get_label_stats(
                streamer=self,
                categorical=categorical,
                numerical=numerical,
                further_stats=further_stats
            )
            n_total_patients += stats["n"]
            for label in all_labels:
                all_vals[label].extend(stats[label]["vals"])

        assert n_total_patients == len(all_file_ids)
        # Compute overall stats
        for label in all_labels:
            all_label_mean[label] = np.mean(all_vals[label])
        for label in numerical:
            all_label_std[label] = np.std(all_vals[label])

        folds = [[] for i in range(k)]  # list of group lists
        fold_means = [{} for i in range(k)]
        fold_it = itertools.cycle(list(range(k)))
        n_groups = len(all_groups)
        used = set([])
        unused = set(list(range(n_groups)))
        fold_target_size = int(n_total_patients / k)
        # Fill folds in round robin fashion
        full_folds = []
        while len(full_folds) < k:
            fold_i = next(fold_it)
            while (fold_i in full_folds):
                fold_i = next(fold_it)

            fold = folds[fold_i]
            fold_mean = fold_means[fold_i]

            if len(fold) == 0:
                # Get one unused group
                idx = unused.pop()  # deterministic for integer hashes
                used.add(idx)
                fold.append(all_groups[idx])
                fold_n = len(all_groups[idx].file_ids)
                first_stats = all_groups[idx].get_label_stats(
                    self, categorical, numerical, further_stats
                )
                for label in all_labels:
                    fold_mean[label] = first_stats[label]["mean"]

                fold_mean["n"] = fold_n

                continue

            if len(unused) == 0:
                break
            # Find best group
            unused_idx = list(unused)
            best_idx = -1
            best_sc = -1
            best_mean = None
            for idx in unused_idx:
                cur = all_groups[idx]
                stats = cur.get_label_stats(
                    self, categorical, numerical, further_stats
                )
                new_mean = copy.deepcopy(fold_mean)
                # Update mean and std
                sc = 0
                new_n = fold_n + stats["n"]
                for label in all_labels:
                    new_mean[label] = \
                        stats[label]["mean"] * (stats["n"] / new_n) \
                        + fold_mean[label] * (fold_n / new_n)
                    s = abs(new_mean[label] - all_label_mean[label])
                    if label in numerical:
                        s /= all_label_std[label]
                    sc += s
                # Compute score
                if (best_sc) == -1 or (best_sc != -1 and best_sc > sc):
                    best_sc = sc
                    best_idx = idx
                    best_mean = copy.deepcopy(new_mean)

            unused.remove(best_idx)
            used.add(best_idx)
            fold.append(all_groups[best_idx])
            # update fold stats and fold_n
            fold_n += len(all_groups[best_idx].file_ids)
            fold_mean = copy.deepcopy(best_mean)
            fold_mean["n"] = fold_n
            fold_means[fold_i] = fold_mean

            if fold_n >= fold_target_size:
                full_folds.append(fold_i)

        # Add unused to last fold
        for idx in unused:
            folds[next(fold_it)].append(all_groups[idx])

        # Set train and test groups
        train_ids = []
        test_ids = []
        test_fold_idx = self.config["test_fold"]
        for i in range(k):
            if i == test_fold_idx:
                for g in folds[i]:
                    test_ids += g.file_ids
            else:
                for g in folds[i]:
                    train_ids += g.file_ids

        # Print stats about folds
        if not self.silent:
            print("###### Folds info #######")
            for i in range(k):
                self.print_stats(folds[i])
            print("#########################")

        # Check that folds are disjoint
        for i in range(k):
            i_ids = []
            for g in folds[i]:
                i_ids += g.file_ids
            i_ids = set(i_ids)
            for j in range(i + 1, k):
                j_ids = []
                for g in folds[j]:
                    j_ids += g.file_ids
                j_ids = set(j_ids)

                assert len(i_ids.intersection(j_ids)) == 0

        return train_ids, test_ids

    def make_balanced_train_test_split(self, patient_groups):
        print("Making balanced split")
        # these should be mutually exclusive
        balanced_labels = self.config["balanced_labels"]
        train_patients = set([])
        test_patients = set([])
        train_ratio = self.config["train_ratio"]

        groups = self.make_one_sample_groups()
        train_ids = []
        test_ids = []

        toggle = True
        for label in balanced_labels:
            # get groups for which this label is set
            cur = []
            for group in groups:
                fid = group.file_ids[0]
                val = self.file_id_to_meta[fid][label]
                if val == 1:
                    cur.append(group)
                    assert group.is_train is None

            # Make sure patients of train and test are mutually disjoint
            train = 0
            test = 0
            remaining = []
            for g in cur:
                patient = self.get_patient_id(g.file_ids[0])
                if patient in train_patients:
                    g.is_train = True
                    train += 1
                    train_ids.append(g.file_ids[0])
                elif patient in test_patients:
                    g.is_train = False
                    test += 1
                    test_ids.append(g.file_ids[0])
                else:
                    remaining.append(g)

            if len(remaining) == 0:
                continue
            # Make equal split
            n_train = int(train_ratio * len(cur))
            n_train -= train
            n_train = max(n_train, 0)

            # Group by patient
            patient_to_groups = OrderedDict()
            for g in remaining:
                patient = self.get_patient_id(g.file_ids[0])
                if patient not in patient_to_groups:
                    patient_to_groups[patient] = []

                patient_to_groups[patient].append(g)

            remaining = list(patient_to_groups.values())

            if self.shuffle:
                self.np_random.shuffle(remaining)

            # balance gender
            gender_0 = []
            gender_1 = []
            for g in remaining:
                fid = g[0].file_ids[0]
                gender = self.get_gender(fid)
                if gender == 0:
                    gender_0.append(g)
                elif gender == 1:
                    gender_1.append(g)

            remaining = merge_list_of_lists_by_size(gender_0, gender_1)

            # Compute split index
            split = len(remaining)
            s = 0
            for i in range(len(remaining)):
                s += len(remaining[i])
                if s >= n_train:
                    split = i
                    break

            if train_ratio == 1:
                split = len(remaining) + 10

            for i, g_list in enumerate(remaining):
                for g in g_list:
                    # Alternatively make train set a bit smaller or
                    # a bit larger than it should be.
                    if toggle:
                        g.is_train = i < split
                    else:
                        g.is_train = i <= split
                    patient = self.get_patient_id(g.file_ids[0])

                    if g.is_train:
                        train_patients = train_patients.union(set([patient]))
                        train_ids += g.file_ids
                    else:
                        test_patients = test_patients.union(set([patient]))
                        test_ids += g.file_ids

            toggle = not toggle

        return train_ids, test_ids

    def make_train_test_split(self, patient_groups):
        if self.config["n_folds"] == 0:
            if not self.silent:
                print(">>>>>> Train-test split")
            return self.make_balanced_train_test_split(patient_groups)
        else:
            if not self.silent:
                print(">>>>>> k-fold split")
            return self.make_balanced_k_fold_split(patient_groups)


class MRISamePatientSameAgePairStream(MRISingleStream):
    def group_data(self, train_ids, test_ids):
        groups = []
        not_found = 0
        # Group by patient, same patient iff same patient_label
        patient_to_file_ids = OrderedDict()
        for file_id in self.file_id_to_meta:
            record = self.file_id_to_meta[file_id]
            if "file_path" in record:
                patient_label = record["patient_label"]
                if patient_label not in patient_to_file_ids:
                    patient_to_file_ids[patient_label] = []
                patient_to_file_ids[patient_label].append(
                    file_id
                )

            else:
                not_found += 1

        if not_found > 0:
            warnings.warn("{} files not found".format(not_found))

        # Collect train and test patients
        train_files = train_ids
        test_files = test_ids

        # Make arbitrary test 
        groups = []
        test_groups = self.produce_groups(test_files, 2, train=False)
        groups += test_groups

        # Get train patient labels
        train_patients = set([])
        for fid in train_files:
            patient = self.get_patient_label(fid)
            train_patients = train_patients.union(set([patient]))

        # Sort by age, then image_label
        train_patients = sorted(list(train_patients))
        for patient_label in train_patients:
            ids = patient_to_file_ids[patient_label]
            id_with_age = list(map(
                lambda x: (x, self.file_id_to_meta[x]["age"]),
                ids
            ))
            s = sorted(id_with_age, key=lambda x: (x[1], x[0]))

            # build pairs
            L = len(s)
            for i in range(1, L):
                id_1 = s[i - 1][0]
                id_2 = s[i][0]
                diag_1 = self.get_diagnose(id_1)
                diag_2 = self.get_diagnose(id_2)
                age_1 = self.get_age(id_1)
                age_2 = self.get_age(id_2)
                if (age_1 == age_2) and (diag_1 == diag_2):
                    assert id_1 != id_2
                    g = Group([id_1, id_2])
                    g.is_train = True
                    g.patient_label = patient_label
                    groups.append(g)

        return groups


class MRISamePatientPairStream(MRISingleStream):
    def group_data(self, train_ids, test_ids):
        n_pairs = self.config["n_pairs"]
        groups = []

        patient_to_file_ids = self.get_patient_to_file_ids_mapping()

        # Remap patient to diagnose to file_ids
        patient_to_diags = OrderedDict()
        for patient_label in patient_to_file_ids:
            file_ids = patient_to_file_ids[patient_label]
            diagnoses = [(fid, self.get_diagnose(fid)) for fid in file_ids]

            dic = OrderedDict()
            for t in diagnoses:
                fid, diag = t
                if diag not in dic:
                    dic[diag] = []
                dic[diag].append(fid)

            patient_to_diags[patient_label] = dic

        # Sample some pairs
        sampled = set()

        def get_common_diag(p1):
            s = list(patient_to_diags[p1].keys())
            possible = []
            for d in s:
                if len(patient_to_diags[p1][d]) > 1:
                    possible.append(d)

            if len(possible) == 0:
                return None

            idx = self.np_random.randint(0, len(possible))
            return possible[idx], possible[idx]

        def get_different_diag(p1):
            s = list(patient_to_diags[p1].keys())

            if len(s) <= 1:
                return None

            # build possible pairs
            x1, x2 = self.np_random.randint(0, len(s), 2)
            while (x1 == x2):
                x1, x2 = self.np_random.randint(0, len(s), 2)

            return s[x1], s[x2]

        def sample_file_ids(ids):
            idx = self.np_random.randint(0, len(ids))
            return ids[idx]

        patient_labels = list(patient_to_diags.keys())
        if self.shuffle:
            self.np_random.shuffle(patient_labels)

        n_patients = len(patient_labels)
        n_train = int(self.config["train_ratio"] * n_patients)
        n_train_pairs = int(self.config["train_ratio"] * n_pairs)
        train_labels = patient_labels[:n_train]
        test_labels = patient_labels[n_train:]
        print("n_train {}".format(n_train))
        if self.config["same_diagnosis"]:
            get_some_diag = get_common_diag
        else:
            get_some_diag = get_different_diag

        def sample_pair(i):
            if i < n_train_pairs:
                n_patients = len(train_labels)
                patient_labels = train_labels
            else:
                n_patients = len(test_labels)
                patient_labels = test_labels
            # sample first patient
            p1 = self.np_random.randint(0, n_patients)
            p1 = patient_labels[p1]
            d = get_some_diag(p1)
            # sample until both patients have a different diagnose
            while (d is None):
                p1 = self.np_random.randint(0, n_patients)
                p1 = patient_labels[p1]
                d = get_some_diag(p1)

            d1, d2 = d
            id1 = sample_file_ids(patient_to_diags[p1][d1])
            id2 = sample_file_ids(patient_to_diags[p1][d2])

            if self.config["same_diagnosis"]:
                while (id2 == id1):
                    id2 = sample_file_ids(patient_to_diags[p1][d2])

            return tuple(sorted((id1, id2)))

        for i in range(n_pairs):
            pair = sample_pair(i)
            while pair in sampled:
                pair = sample_pair(i)

            sampled.add(pair)

        sampled = sorted(list(sampled))
        for i, s in enumerate(sampled):
            lll = s
            id1 = lll[0]
            id2 = lll[1]
            groups.append(Group([id1, id2]))
            groups[-1].is_train = (i < n_train_pairs)
            assert self.get_patient_label(id1) == self.get_patient_label(id2)
            if self.config["same_diagnosis"]:
                assert self.get_diagnose(id1) == self.get_diagnose(id2)
            else:
                assert self.get_diagnose(id1) != self.get_diagnose(id2)

        return groups


class MRIDifferentPatientPairStream(MRISingleStream):
    def group_data(self, train_ids, test_ids):
        n_pairs = self.config["n_pairs"]
        groups = []

        patient_to_file_ids = self.get_patient_to_file_ids_mapping()

        # Remap patient to diagnose to file_ids
        patient_to_diags = OrderedDict()
        for patient_label in patient_to_file_ids:
            file_ids = patient_to_file_ids[patient_label]
            diagnoses = [(fid, self.get_diagnose(fid)) for fid in file_ids]

            dic = OrderedDict()
            for t in diagnoses:
                fid, diag = t
                if diag not in dic:
                    dic[diag] = []
                dic[diag].append(fid)

            patient_to_diags[patient_label] = dic

        # Sample some pairs
        sampled = set()

        def get_common_diag(p1, p2):
            s1 = set(list(patient_to_diags[p1].keys()))
            s2 = set(list(patient_to_diags[p2].keys()))
            inter = s1.intersection(s2)
            inter = sorted(list(inter))

            if len(inter) == 0:
                return None

            idx = self.np_random.randint(0, len(inter))
            return inter[idx], inter[idx]

        def get_different_diag(p1, p2):
            s1 = sorted(list(patient_to_diags[p1].keys()))
            s2 = sorted(list(patient_to_diags[p2].keys()))

            # build possible pairs
            pairs = list(itertools.product(s1, s2))
            self.np_random.shuffle(pairs)
            for d1, d2 in pairs:
                if d1 != d2:
                    return d1, d2

            return None

        def sample_file_ids(ids):
            idx = self.np_random.randint(0, len(ids))
            return ids[idx]

        patient_labels = sorted(list(patient_to_diags.keys()))
        if self.shuffle:
            self.np_random.shuffle(patient_labels)

        n_patients = len(patient_labels)
        n_train = int(self.config["train_ratio"] * n_patients)
        n_train_pairs = int(self.config["train_ratio"] * n_pairs)
        train_labels = patient_labels[:n_train]
        test_labels = patient_labels[n_train:]
        print("n_train {}".format(n_train))
        if self.config["same_diagnosis"]:
            get_some_diag = get_common_diag
        else:
            get_some_diag = get_different_diag

        def sample_pair(i):
            if i < n_train_pairs:
                n_patients = len(train_labels)
                patient_labels = train_labels
            else:
                n_patients = len(test_labels)
                patient_labels = test_labels
            # sample first patient
            p1 = self.np_random.randint(0, n_patients)
            p1 = patient_labels[p1]
            # sample second patient
            p2 = self.np_random.randint(0, n_patients)
            p2 = patient_labels[p2]
            d = get_some_diag(p1, p2)
            # sample until both patients have a different diagnose
            while (d is None):
                p2 = self.np_random.randint(0, n_patients)
                p2 = patient_labels[p2]
                d = get_some_diag(p1, p2)

            d1, d2 = d
            id1 = sample_file_ids(patient_to_diags[p1][d1])
            id2 = sample_file_ids(patient_to_diags[p2][d2])
            return tuple(sorted((id1, id2)))

        for i in range(n_pairs):
            pair = sample_pair(i)
            while pair in sampled:
                pair = sample_pair(i)

            sampled.add(pair)

        sampled = sorted(list(sampled))
        for i, s in enumerate(sampled):
            lll = s
            id1 = lll[0]
            id2 = lll[1]
            groups.append(Group([id1, id2]))
            groups[-1].is_train = (i < n_train_pairs)
            if self.config["same_diagnosis"]:
                assert self.get_diagnose(id1) == self.get_diagnose(id2)
            else:
                assert self.get_diagnose(id1) != self.get_diagnose(id2)

        return groups


class MRIDiagnosePairStreamDeprecated(MRISingleStream):
    """
    Allows for more exact sampling wrt. diagnoses. A diagnosis
    pairs is expected in the config file under 'diagnoses'. For
    every sampled pair, there is one image for each of diagnoses
    specified. One can also specify if both images should belong
    to the same patient or not.
    """    
    def get_diagnoses(self):
        return self.config["diagnoses"]

    def group_data(self, train_ids, test_ids):
        self.start_time = process_time()
        n_pairs = self.config["n_pairs"]
        n_train_pairs = int(self.config["train_ratio"] * n_pairs)

        # Collect train and test patients
        train_files = train_ids
        test_files = test_ids

        train_labels = []
        for fid in train_files:
            label = self.get_patient_label(fid)
            train_labels.append(label)
        train_labels = set(train_labels)

        test_labels = []
        for fid in test_files:
            label = self.get_patient_label(fid)
            test_labels.append(label)
        test_labels = set(test_labels)

        assert len(train_labels.intersection(test_labels)) == 0

        # Make arbitrary test 
        groups = []

        patient_to_file_ids = self.get_patient_to_file_ids_mapping()

        # Map diagnosis to patient to file_ids
        diag_to_patient = OrderedDict()
        for patient_label in patient_to_file_ids:
            file_ids = patient_to_file_ids[patient_label]
            diagnoses = [(fid, self.get_diagnose(fid)) for fid in file_ids]

            for fid, d in diagnoses:
                if d not in diag_to_patient:
                    diag_to_patient[d] = OrderedDict()
                if patient_label not in diag_to_patient[d]:
                    diag_to_patient[d][patient_label] = []
                diag_to_patient[d][patient_label].append(fid)

        diagnoses_to_sample = self.config["diagnoses"]
        assert len(diagnoses_to_sample) == 2
        self.config["same_diagnosis"] = \
            (diagnoses_to_sample[0] == diagnoses_to_sample[1])

        def check_time():
            delta_t = process_time() - self.start_time
            return delta_t <= 5

        def sample_patient(diagnosis, patient_labels):
            patients = set(diag_to_patient[diagnosis].keys())
            sample_from = set(patient_labels).intersection(patients)
            sample_from = sorted(list(sample_from))
            n = len(sample_from)
            idx = self.np_random.randint(0, n)
            return sample_from[idx]

        def sample_file_ids(p1, d1, p2, d2):
            def _sample(d, p):
                fids = diag_to_patient[d][p]
                n = len(fids)
                idx = self.np_random.randint(0, n)
                return fids[idx]

            return _sample(d1, p1), _sample(d2, p2)

        def get_pair_sample(patient_labels):
            d1, d2 = diagnoses_to_sample
            p1 = sample_patient(d1, patient_labels)

            if self.config["same_patient"] and self.config["same_diagnosis"]:
                while (len(diag_to_patient[d1][p1]) < 2 and check_time()):
                    p1 = sample_patient(d1, patient_labels)

                p2 = p1
            elif self.config["same_patient"]:
                while (not(p1 in diag_to_patient[d1] and
                       p1 in diag_to_patient[d2]) and check_time()):
                    p1 = sample_patient(d1, patient_labels)

                p2 = p1
            else:
                # different patient
                p2 = sample_patient(d2, patient_labels)
                while (p2 == p1 and check_time()):
                    p2 = sample_patient(d2, patient_labels)

            if not check_time():
                return None

            # sample file ids
            fid1, fid2 = sample_file_ids(p1, d1, p2, d2)
            while (fid1 == fid2):
                fid1, fid2 = sample_file_ids(p1, d1, p2, d2)

            assert self.get_diagnose(fid1) == d1
            assert self.get_diagnose(fid2) == d2
            return fid1, fid2

        def sample_pair(i):
            if i < n_train_pairs:
                patient_labels = train_labels
            else:
                patient_labels = test_labels

            p = get_pair_sample(patient_labels)

            if p is None:
                return None

            # sort pair by age
            age1 = self.get_age(p[0])
            age2 = self.get_age(p[1])
            if age1 > age2:
                p = p[::-1]

            if i < n_train_pairs:
                p = tuple([p[0], p[1], True])
            else:
                p = tuple([p[0], p[1], False])

            return p

        sampled = set([])
        for i in range(n_pairs):
            pair = sample_pair(i)
            while pair in sampled:
                if pair is None:
                    break
                pair = sample_pair(i)

            if pair is None:
                break
            sampled.add(pair)

        if len(sampled) < n_pairs:
            warnings.warn("{} sampled only {} pairs!!!".format(self.__class__.__name__, len(sampled)))

        sampled = sorted(list(sampled))
        for i, s in enumerate(sampled):
            lll = s
            id1 = lll[0]
            id2 = lll[1]
            groups.append(Group([id1, id2]))
            groups[-1].is_train = lll[2]
            if self.config["same_diagnosis"]:
                assert self.get_diagnose(id1) == self.get_diagnose(id2)
            else:
                assert self.get_diagnose(id1) != self.get_diagnose(id2)

            assert (self.get_patient_id(id1) == self.get_patient_id(id2)) == \
                self.config["same_patient"]

        return groups


class MRIDiagnosePairStream(MRISingleStream):
    """
    Allows for more exact sampling wrt. diagnoses. A diagnosis
    pairs is expected in the config file under 'diagnoses'. For
    every sampled pair, there is one image for each of diagnoses
    specified. One can also specify if both images should belong
    to the same patient or not.
    """
    def get_diagnoses(self):
        return self.config["diagnoses"]

    def distinct_pairs(self, a, b):
        for p in itertools.product(a, b):
            if p[0] != p[1]:
                yield p

    def same_patient_gen(self, file_ids):
        # patient to pair gen
        diagnoses = self.get_diagnoses()

        patient_groups = self.make_patient_groups(file_ids)
        patient_to_gen = {}
        pids = []
        for g in patient_groups:
            pid = self.get_patient_id(g.file_ids[0])
            pids.append(pid)
            # first images
            first = [fid for fid in g.file_ids
                     if self.get_diagnose(fid) == diagnoses[0]]
            # second images
            second = [fid for fid in g.file_ids
                      if self.get_diagnose(fid) == diagnoses[1]]

            self.np_random.shuffle(first)
            self.np_random.shuffle(second)

            if len(first) == 0 or len(second) == 0:
                continue

            if diagnoses[0] != diagnoses[1]:
                patient_to_gen[pid] = self.distinct_pairs(first, second)
            else:
                combos = list(itertools.combinations(first, 2))
                self.np_random.shuffle(combos)
                patient_to_gen[pid] = iter(combos)

        done = False
        pids = list(patient_to_gen.keys())
        while not done:
            self.np_random.shuffle(pids)
            done = True
            for pid in pids:
                gen = patient_to_gen[pid]
                try:
                    pair = next(gen)
                    if pair is not None:
                        done = False
                        yield pair
                except StopIteration:
                    continue

    def different_patient_gen(self, file_ids):
        # patient to pair gen
        diagnoses = self.get_diagnoses()

        patient_groups = self.make_patient_groups(file_ids)
        pids = []
        pid_to_fids = {}
        for g in patient_groups:
            pid = self.get_patient_id(g.file_ids[0])
            # print("{} has {} images".format(pid, len(g.file_ids)))
            pids.append(pid)
            pid_to_fids[pid] = g.file_ids

        if diagnoses[0] == diagnoses[1]:
            patient_pairs = list(itertools.combinations(pids, 2))
        else:
            patient_pairs = list(self.distinct_pairs(pids, pids))

        pair_to_gen = {}
        for pid1, pid2 in patient_pairs:
            first = [fid for fid in pid_to_fids[pid1]
                     if self.get_diagnose(fid) == diagnoses[0]]
            # second images
            second = [fid for fid in pid_to_fids[pid2]
                      if self.get_diagnose(fid) == diagnoses[1]]

            # print("{} has {} images with diag {}".format(pid1, len(first), diagnoses[0]))
            # print("{} has {} images with diag {}".format(pid2, len(second), diagnoses[1]))
            self.np_random.shuffle(first)
            self.np_random.shuffle(second)

            p = tuple([pid1, pid2])
            combos = list(itertools.product(first, second))
            self.np_random.shuffle(combos)
            if len(combos) > 0:
                pair_to_gen[p] = iter(combos)

        done = False
        patient_pairs = list(pair_to_gen.keys())
        while not done:
            self.np_random.shuffle(patient_pairs)
            done = True
            for pid1, pid2 in patient_pairs:
                p = tuple([pid1, pid2])
                try:
                    pair = next(pair_to_gen[p])
                    done = False
                    yield pair
                except StopIteration:
                    continue

    def get_pair_gen(self, file_ids):
        if self.config["same_patient"]:
            return self.same_patient_gen(file_ids)
        else:
            return self.different_patient_gen(file_ids)

    def group_data(self, train_ids, test_ids):
        self.start_time = process_time()
        n_pairs = self.config["n_pairs"]
        n_train_pairs = int(self.config["train_ratio"] * n_pairs)

        # Collect train and test patients

        # Make arbitrary test 
        groups = []
        test_groups = self.produce_groups(test_ids, 2, train=False)
        groups += test_groups

        diagnoses_to_sample = self.config["diagnoses"]
        assert len(diagnoses_to_sample) == 2
        self.config["same_diagnosis"] = \
            (diagnoses_to_sample[0] == diagnoses_to_sample[1])

        pair_gen = self.get_pair_gen(train_ids)
        sampled = []

        for pair in pair_gen:
            sampled.append(pair)

            if len(sampled) >= n_train_pairs:
                break

        if len(sampled) < n_train_pairs:
            warnings.warn("{} sampled only {} pairs!!!".format(self.__class__.__name__, len(sampled)))

        sampled = sorted(list(sampled))
        for i, s in enumerate(sampled):
            lll = s
            id1 = lll[0]
            id2 = lll[1]
            groups.append(Group([id1, id2]))
            groups[-1].is_train = True
            if self.config["same_diagnosis"]:
                assert self.get_diagnose(id1) == self.get_diagnose(id2)
            else:
                assert self.get_diagnose(id1) != self.get_diagnose(id2)

            assert (self.get_patient_id(id1) == self.get_patient_id(id2)) == \
                self.config["same_patient"]

        return groups


class SimilarPairStream(MRISingleStream):
    """
    For now, two images are considered as similar if they
    have the same diagnosis.
    """
    def get_diagnoses(self):
        """
        Returns a list of strings designating diagnoses.
        """
        return self.config["diagnoses"]

    def get_max_train_pairs(self):
        return self.config["max_train_pairs"]

    def sampling_iterator(self, file_ids):
        # Group fids by patient
        patient_to_iterator = OrderedDict()

        patient_to_fids = OrderedDict()
        for fid in file_ids:
            p = self.get_patient_id(fid)
            if p not in patient_to_fids:
                patient_to_fids[p] = []
            patient_to_fids[p].append(fid)

        patients = list(patient_to_fids.keys())
        # all file_ids sorted by patient
        patient_to_range = OrderedDict()
        all_fids = []
        for p in patients:
            p_fids = patient_to_fids[p]
            patient_to_iterator[p] = itertools.cycle(p_fids)
            s_idx = len(all_fids)
            all_fids += p_fids
            e_idx = len(all_fids)
            patient_to_range[p] = (s_idx, e_idx)

        for p in itertools.cycle(patients):
            fid = next(patient_to_iterator[p])
            s_idx, e_idx = patient_to_range[p]
            others = all_fids[0: s_idx] + all_fids[e_idx:]
            # Sample from others
            idx = self.np_random.randint(0, len(others))
            yield fid, others[idx]

    def group_data(self, train_ids, test_ids):
        max_train_pairs = self.get_max_train_pairs()
        diagnoses = self.get_diagnoses()

        train_files = train_ids
        test_files = test_ids

        # Make arbitrary test 
        groups = []
        test_groups = self.produce_groups(test_files, 2, train=False)
        groups += test_groups

        # Sample train pairs
        # Group train pairs by diagnose
        diag_to_fids = OrderedDict()
        for d in diagnoses:
            diag_to_fids[d] = []

        if self.shuffle:
            self.np_random.shuffle(train_files)

        for fid in train_files:
            d = self.get_diagnose(fid)
            if d in diag_to_fids:
                diag_to_fids[d].append(fid)

        # Sample same number of pairs per diagnose
        for d in diagnoses:
            fids = diag_to_fids[d]
            d_train = []
            for p in self.sampling_iterator(fids):
                # Check if we have already enough pairs
                if max_train_pairs >= 0:
                    if len(d_train) >= max_train_pairs / len(diagnoses):
                        break

                patient1 = self.get_patient_label(p[0])
                patient2 = self.get_patient_label(p[1])

                if patient1 != patient2:
                    g = Group(list(p))
                    g.is_train = True
                    d_train.append(g)

            groups += d_train

        return groups


class Patient(object):
    def __init__(self, file_ids, patient_id):
        self.file_ids = file_ids
        self.patient_id = patient_id
        self.similar = set([])
        self.dissimilar = set([])
        self.sim_count = 0
        self.dissim_count = 0

        r = np.random.RandomState(11)
        r.shuffle(self.file_ids)
        self.fid_cycle = itertools.cycle(self.file_ids)

    def set_diagnosis(self, diag):
        self.diagnosis = diag

    def get_diagnosis(self):
        return self.diagnosis

    def set_diag_to_patients(self, diag_to_patients):
        self.diag_to_patients = diag_to_patients

    def get_next_image(self):
        return next(self.fid_cycle)

    def find_similar(self, same_patient=False):
        """
        Returns true iff a matching patient was found.
        """
        candidates = self.diag_to_patients[self.diagnosis]
        best_cand = None
        best_size = -1
        if same_patient:
            best_cand = self
        else:
            for cand in candidates:
                if cand.patient_id == self.patient_id:
                    continue
                # allow some size tolerance
                if cand.patient_id not in self.similar:
                    if (best_cand is None) or (len(cand.similar) < best_size):
                        best_size = len(cand.similar)
                        best_cand = cand

        # Found match
        if best_cand is not None:
            self.similar.add(best_cand.patient_id)
            self.sim_count += 1
            if self != best_cand:
                best_cand.similar.add(self.patient_id)
                best_cand.sim_count += 1
            return True

        for c in candidates:
            print("{}: {}".format(c.patient_id, c.similar))
        return False

    def find_dissimilar(self):
        """
        Returns true iff a matching patient was found.
        """
        diagnoses = list(self.diag_to_patients.keys())
        if self.diagnosis == diagnoses[0]:
            candidates = self.diag_to_patients[diagnoses[1]]
        else:
            candidates = self.diag_to_patients[diagnoses[0]]

        best_cand = None
        best_size = -1
        for cand in candidates:
            if cand.patient_id == self.patient_id:
                continue
            # allow some size tolerance
            if cand.patient_id not in self.dissimilar:
                if (best_cand is None) or (len(cand.dissimilar) < best_size):
                    best_size = len(cand.dissimilar)
                    best_cand = cand

        # Found match
        if best_cand is not None:
            self.dissimilar.add(best_cand.patient_id)
            best_cand.dissimilar.add(self.patient_id)
            return True

        for c in candidates:
            print("{}: {}".format(c.patient_id, c.similar))
        return False


class MixedPairStream(MRISingleStream):
    """
    Remove patients from training that have multiple
    diagnoses.
    """
    def get_diagnoses(self):
        return self.config["diagnoses"]

    def get_number_pairs_per_patient(self):
        # The number of pairs per patient
        return self.config["n_patient_pairs"]

    def group_data(self, train_ids, test_ids):
        n_pairs_per_patient = self.get_number_pairs_per_patient()
        diagnoses = self.get_diagnoses()

        patient_groups = self.make_patient_groups(train_ids)

        # Build patients
        all_patients = []
        diag_to_patients = OrderedDict()
        for d in diagnoses:
            diag_to_patients[d] = []

        multiple_diags = 0
        patient_id_to_obj = {}
        for g in patient_groups:
            pat = Patient(
                file_ids=g.file_ids,
                patient_id=self.get_patient_id(g.file_ids[0])
            )
            patient_id_to_obj[pat.patient_id] = pat
            diags = [self.get_diagnose(fid) for fid in g.file_ids]
            # Remove patients with multiple diagnoses
            if len(set(diags)) > 1:
                multiple_diags += 1
                continue  # more than one diagnosis

            pat.set_diagnosis(diags[0])
            pat.set_diag_to_patients(diag_to_patients)
            all_patients.append(pat)
            # Map diagnoses to patients
            diag_to_patients[diags[0]].append(pat)

        if not self.silent:
            print("{} patients with multiple diagnoses".format(multiple_diags))
        # Build pairs, shuffle list of patients for each iteration
        # Pair patients that have the same number of pairs

        # Similar pairs
        for diag in diagnoses:
            patients = diag_to_patients[diag]
            for i in range(n_pairs_per_patient):
                self.np_random.shuffle(patients)
                for pat in patients:
                    if len(pat.similar) == i:
                        assert pat.find_similar(self.config["same_patient"])

        # Dissimilar pairs
        assert len(diagnoses) == 2
        for i in range(n_pairs_per_patient):
            self.np_random.shuffle(diag_to_patients[diagnoses[0]])
            self.np_random.shuffle(diag_to_patients[diagnoses[1]])
            self.np_random.shuffle(all_patients)

            for pat in all_patients:
                if len(pat.dissimilar) == i:
                    assert pat.find_dissimilar()

        # Check that every patient has enough pairs
        for pat in all_patients:
            # allow some tolerance in case patients cannot match up
            # perfectly (dependent on the desired number of pairs)
            assert pat.sim_count >= n_pairs_per_patient
            assert len(pat.dissimilar) >= n_pairs_per_patient

        if not self.silent:
            # Print some stats
            similar_pairs_hist = OrderedDict()
            dissimilar_pairs_hist = OrderedDict()

            for pat in all_patients:
                c = len(pat.similar)
                if c not in similar_pairs_hist:
                    similar_pairs_hist[c] = 1
                else:
                    similar_pairs_hist[c] += 1

                c = len(pat.dissimilar)
                if c not in dissimilar_pairs_hist:
                    dissimilar_pairs_hist[c] = 1
                else:
                    dissimilar_pairs_hist[c] += 1

            print("Similar pair stats")
            for k, v in similar_pairs_hist.items():
                print(">>> Number patients with {} pairs: {}".format(k, v))

            print("Dissimilar pair stats")
            for k, v in dissimilar_pairs_hist.items():
                print(">>> Number patients with {} pairs: {}".format(k, v))

        train_pairs = set()
        # Avoid duplicate pairs
        for pat in all_patients:
            for other in pat.similar.union(pat.dissimilar):
                tup = tuple([pat.patient_id, other])
                if tup[0] > tup[1]:
                    tup = tuple([tup[1], tup[0]])
                train_pairs.add(tup)

        train_groups = []
        train_pairs = sorted(list(train_pairs))
        self.np_random.shuffle(train_pairs)
        for tup in train_pairs:
            # Given pair of patient ids, we need to sample images
            pat1 = patient_id_to_obj[tup[0]]
            pat2 = patient_id_to_obj[tup[1]]
            if self.config["same_patient"] and pat1 == pat2:
                for i in range(pat1.sim_count):
                    g = Group([pat1.get_next_image(), pat2.get_next_image()], True)
                    train_groups.append(g)
            else:
                g = Group([pat1.get_next_image(), pat2.get_next_image()], True)
                train_groups.append(g)
        test_groups = self.produce_groups(test_ids, 2, train=False)

        # Make sure all patients are used
        used_patients = set([])
        for g in train_groups:
            for fid in g.file_ids:
                pat_id = self.get_patient_id(fid)
                used_patients.add(pat_id)

        init_patients = set([pat.patient_id for pat in all_patients])
        assert used_patients == init_patients

        return train_groups + test_groups


class AnyPairStream(MRISingleStream):
    def group_data(self, train_ids, test_ids):
        train_files = train_ids
        test_files = test_ids

        # Avoid that pairs are too similari due to ordering
        # in csv file
        self.np_random.shuffle(train_files)
        self.np_random.shuffle(test_files)

        train_groups = self.produce_groups(train_files, 2, train=True)
        test_groups = self.produce_groups(test_files, 2, train=False)

        return train_groups + test_groups


class MRIConversionSingleStream(MRISingleStream):
    def select_file_ids(self, file_ids):
        return self.select_conversion_file_ids(file_ids)


class MRIClassifierSingleStream(MRISingleStream):
    def select_file_ids(self, file_ids):
        file_ids = [
            f for f in file_ids if
            self.get_meta_info_by_key(f, "n_conversions") == 0 and
            self.get_diagnose(f) in self.config["use_diagnoses"]
        ]

        return file_ids


class MRISplitInfoPrinter(MRISingleStream):
    def __init__(self, *args, **kwargs):
        super(MRISplitInfoPrinter, self).__init__(
            *args,
            **kwargs
        )

        exit()


class SplitStats(object):
    def __init__(self, image_count, subject_count, age_mean, age_std):
        self.image_count = image_count
        self.subject_count = subject_count
        self.age_mean = age_mean
        self.age_std = age_std

    def get_metrics(self):
        names = ["image_count", "subject_count", "age_mean", "age_std"]
        values = [self.image_count, self.subject_count, self.age_mean,
                  self.age_std]

        return names, values


class MRICVTable(MRISingleStream):
    def __init__(self, *args, **kwargs):
        super(MRICVTable, self).__init__(
            *args,
            **kwargs
        )

        self.cv_split_folder = self.config["cv_split_folder"]
        self.print_cv_table()
        exit()

    def print_cv_table(self):
        # find all splits and get stats
        all_paths = []
        names = os.listdir(self.cv_split_folder)
        for name in names:
            p = os.path.join(self.cv_split_folder, name)
            if os.path.isdir(p):
                all_paths.append(p)

        all_paths = sorted(all_paths)
        all_stats = []

        for p in all_paths:
            stats = self.get_stats(os.path.join(p, "test.txt"))
            all_stats.append(stats)

        print(all_paths)
        key_to_vals = OrderedDict()
        for stat in all_stats:
            for d, d_stat in stat.items():
                met_names, met_vals = d_stat.get_metrics()
                for name, val in zip(met_names, met_vals):
                    k = d + "_" + name
                    if k not in key_to_vals:
                        key_to_vals[k] = []

                    key_to_vals[k].append(val)

        columns = [values for values in key_to_vals.values()]
        header = key_to_vals.keys()

        df = pd.DataFrame(
            data=np.array(columns).T,
            columns=header
        )

        print(df.round(4).to_latex(index=False))

    def get_stats(self, file_path):
        with open(file_path, 'r') as f:
            fids = [line.strip() for line in f]

        diag_file_ids = {
            d: []
            for d in self.config["use_diagnoses"]
        }
        
        if self.config["conversion"]:
            # keep only t0 images
            keep = []
            patient_groups = self.make_patient_groups(fids)
            for g in patient_groups:
                ss = sorted(g.file_ids, key=lambda x: self.get_exact_age(x))
                keep.append(ss[0])
            fids = keep

        for fid in fids:
            d = self.get_diagnose(fid)
            diag_file_ids[d].append(fid)

        diag_to_stat = OrderedDict()
        for d in self.config["use_diagnoses"]:
            fids = diag_file_ids[d]
            image_count = len(fids)
            subject_count = len(self.make_patient_groups(fids))
            ages = [self.get_exact_age(fid) for fid in fids]
            age_mean = np.mean(ages)
            age_std = np.std(ages)
            diag_to_stat[d] = SplitStats(
                image_count=image_count,
                subject_count=subject_count,
                age_mean=age_mean,
                age_std=age_std
            )

        return diag_to_stat
