import nibabel as nib
import warnings
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import itertools

from .base import FileStream
from .base import Group


class MRIImageLoader(object):
    def load_image(self, image_path):
        mri_image = nib.load(image_path)
        mri_image = mri_image.get_data()

        return mri_image


class MRISingleStream(FileStream, MRIImageLoader):
    """
    Stream files one by one.
    """
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

        return batches

    def group_data(self):
        # We just to stream the images one by one
        groups = []
        for key in self.file_id_to_meta:
            if "file_path" in self.file_id_to_meta[key]:
                g = Group([key])
                g.patient_label = self.get_patient_label(key)
                groups.append(g)

        return groups

    def load_sample(self, file_path):
        im = self.load_image(file_path)
        if self.config["downsample"]["enabled"]:
            # im = im / np.max(im)
            shape = tuple(self.config["downsample"]["shape"])
            # im = resize(im, shape, anti_aliasing=False)
            im = np.zeros(shape)

        return im

    def make_train_test_split(self):
        # pairs belonging to the same person should be in same split
        train_ratio = self.config["train_ratio"]
        self.np_random.shuffle(self.groups)
        # Make split with respect to patients
        patient_labels = list(set([g.patient_label for g in self.groups]))
        n_train = int(train_ratio * len(patient_labels))
        train_labels = set(patient_labels[:n_train])

        for g in self.groups:
            g.is_train = (g.patient_label in train_labels)


class MRISamePatientSameAgePairStream(MRISingleStream):
    def group_data(self):
        groups = []
        not_found = 0
        # Group by patient, same patient iff same patient_label
        patient_to_file_ids = {}
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
        # Sort by age, then image_label
        for patient_label in patient_to_file_ids:
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
                    g.patient_label = patient_label
                    groups.append(g)

        return groups


class MRISamePatientPairStream(MRISingleStream):
    def group_data(self):
        n_pairs = self.config["n_pairs"]
        groups = []

        patient_to_file_ids = self.get_patient_to_file_ids_mapping()

        # Remap patient to diagnose to file_ids
        patient_to_diags = {}
        for patient_label in patient_to_file_ids:
            file_ids = patient_to_file_ids[patient_label]
            diagnoses = [(fid, self.get_diagnose(fid)) for fid in file_ids]

            dic = {}
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

    def make_train_test_split(self):
        # split is done during sampling
        pass


class MRIDifferentPatientPairStream(MRISingleStream):
    def group_data(self):
        n_pairs = self.config["n_pairs"]
        groups = []

        patient_to_file_ids = self.get_patient_to_file_ids_mapping()

        # Remap patient to diagnose to file_ids
        patient_to_diags = {}
        for patient_label in patient_to_file_ids:
            file_ids = patient_to_file_ids[patient_label]
            diagnoses = [(fid, self.get_diagnose(fid)) for fid in file_ids]

            dic = {}
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
            inter = list(inter)

            if len(inter) == 0:
                return None

            idx = self.np_random.randint(0, len(inter))
            return inter[idx], inter[idx]

        def get_different_diag(p1, p2):
            s1 = list(patient_to_diags[p1].keys())
            s2 = list(patient_to_diags[p2].keys())

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

    def make_train_test_split(self):
        # split is done during sampling
        pass
