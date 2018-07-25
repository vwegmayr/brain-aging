import nibabel as nib
import warnings
import numpy as np
import itertools
from time import process_time
from collections import OrderedDict
import os
import copy

from .base import FileStream
from .base import Group


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
        self.normalization_computed = False
        self.normalize_images = self.config["normalize_images"]
        if self.normalize_images:
            self.compute_image_normalization()

    def get_batches(self, train=True):
        groups = [group for group in self.groups
                  if group.is_train == train]

        if self.shuffle:
            self.np_random.shuffle(groups)

        if self.batch_size == -1:
            return [[group for group in groups]]

        n_samples = len(groups)
        n_batches = int(n_samples / self.batch_size)

        batches = []
        for i in range(n_batches):
            bi = i * self.batch_size
            ei = (i + 1) * self.batch_size
            batches.append(groups[bi:ei])

        if n_batches * self.batch_size < n_samples:
            bi = n_batches * self.batch_size
            batches.append(groups[bi:])

        n_groups = 0
        for batch in batches:
            n_groups += len(batch)
        assert n_groups == len(groups)
        return batches

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
        file_ids = self.all_file_ids.difference(test_ids)
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
            im = self.load_image(p)
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
        std = np.std(im)
        if np.isclose(0, std):
            std = 1
        im = (im - np.mean(im)) / std
        im = (im - self.voxel_means) / self.voxel_stds
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

        self.groups = None
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

        self.groups = None
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

        self.groups = None
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

    def find_similar(self):
        """
        Returns true iff a matching patient was found.
        """
        candidates = self.diag_to_patients[self.diagnosis]
        best_cand = None
        best_size = -1
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
            best_cand.similar.add(self.patient_id)
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
                        assert pat.find_similar()

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
            assert len(pat.similar) >= n_pairs_per_patient
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


class BatchProvider(object):
    def __init__(self, streamer, file_ids, label_key, prefetch=1000):
        self.file_ids = file_ids
        assert len(file_ids) > 0
        self.streamer = streamer
        self.label_key = label_key
        self.prefetch = prefetch
        self.loaded = []
        self.np_random = np.random.RandomState(seed=11)
        self.fid_gen = self.next_fid()
        self.img_gen = self.next_image()

    def next_fid(self):
        self.np_random.shuffle(self.file_ids)
        p = 0
        while (1):
            if p < len(self.file_ids):
                yield self.file_ids[p]
                p += 1
            else:
                p = 0
                self.np_random.shuffle(self.file_ids)

    def next_image(self):
        loaded = []

        while (1):
            if len(loaded) == 0:
                # prefetch
                for i in range(self.prefetch):
                    fid = next(self.fid_gen)
                    p = self.streamer.get_file_path(fid)
                    im = self.streamer.load_sample(p)
                    if self.streamer.normalize_images:
                        im = self.streamer.normalize_image(im)
                    label = self.streamer.get_meta_info_by_key(
                        fid, self.label_key
                    )
                    loaded.append([im, label])
            else:
                el = loaded[0]
                loaded.pop(0)
                yield el

    def next_batch(self, batch_size):
        X_batch = []
        y_batch = []
        for i in range(batch_size):
            x, y = next(self.img_gen)
            X_batch.append(x)
            y_batch.append(y)

        return np.array(X_batch), np.array(y_batch)


class AnySingleStream(MRISingleStream):
    """
    Compatibility with Baumgartner VAGAN implementation.
    """
    def __init__(self, *args, **kwargs):
        super(AnySingleStream, self).__init__(
            *args,
            **kwargs
        )
        self.AD_key = self.config["AD_key"]  
        self.CN_key = self.config["CN_key"]  # Control group
        self.set_up_batches()

    def get_ad_cn_ids(self, file_ids):
        ad_ids = []
        cn_ids = []

        for fid in file_ids:
            v = self.get_meta_info_by_key(fid, self.AD_key)
            if v == 1:
                ad_ids.append(fid)
            else:
                cn_ids.append(fid)

        return ad_ids, cn_ids

    def set_up_batches(self):
        # Train batches
        train_ids = self.get_train_ids()
        train_AD_ids, train_CN_ids = self.get_ad_cn_ids(train_ids)
        self.trainAD = BatchProvider(
            streamer=self,
            file_ids=train_AD_ids,
            label_key=self.AD_key,
            prefetch=self.config["prefetch"]
        )
        self.trainCN = BatchProvider(
            streamer=self,
            file_ids=train_CN_ids,
            label_key=self.CN_key,
            prefetch=self.config["prefetch"]
        )

        # Validation batches
        validation_ids = self.get_validation_ids()
        valid_AD_ids, valid_CN_ids = self.get_ad_cn_ids(validation_ids)
        self.validationAD = BatchProvider(
            streamer=self,
            file_ids=valid_AD_ids,
            label_key=self.AD_key,
            prefetch=self.config["prefetch"]
        )
        self.validationCN = BatchProvider(
            streamer=self,
            file_ids=valid_CN_ids,
            label_key=self.CN_key,
            prefetch=self.config["prefetch"]
        )

        # Test batches
        test_ids = self.get_test_ids()
        test_AD_ids, test_CN_ids = self.get_ad_cn_ids(test_ids)
        self.testAD = BatchProvider(
            streamer=self,
            file_ids=test_AD_ids,
            label_key=self.AD_key,
            prefetch=self.config["prefetch"]
        )
        self.testCN = BatchProvider(
            streamer=self,
            file_ids=test_CN_ids,
            label_key=self.CN_key,
            prefetch=self.config["prefetch"]
        )
