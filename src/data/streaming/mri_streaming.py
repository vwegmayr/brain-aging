import nibabel as nib
import warnings
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import itertools
from time import process_time
from collections import OrderedDict

from .base import FileStream
from .base import Group


class MRIImageLoader(object):
    def load_image(self, image_path):
        mri_image = nib.load(image_path)
        mri_image = mri_image.get_data()

        return mri_image.astype(np.float16)


class MRISingleStream(FileStream, MRIImageLoader):
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

    def make_balanced_train_test_split(self):
        print("Making balanced split")
        # these should be mutually exclusive
        balanced_labels = self.config["balanced_labels"]
        train_patients = set([])
        test_patients = set([])
        train_ratio = self.config["train_ratio"]

        toggle = True
        for label in balanced_labels:
            # get groups for which this label is set
            cur = []
            for group in self.groups:
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
                elif patient in test_patients:
                    g.is_train = False
                    test += 1
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
                    else:
                        test_patients = test_patients.union(set([patient]))

            toggle = not toggle

    def make_train_test_split(self):
        return self.make_balanced_train_test_split()


class MRISamePatientSameAgePairStream(MRISingleStream):
    def group_data(self):
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

        self.groups = self.make_one_sample_groups()
        self.make_balanced_train_test_split()

        # Collect train and test patients
        train_files = [g.file_ids[0] for g in self.groups
                       if g.is_train == True]
        test_files = [g.file_ids[0] for g in self.groups
                      if g.is_train == False]

        self.groups = None
        # Make arbitrary test 
        groups = []
        test_groups = self.produce_test_groups(test_files, 2)
        groups += test_groups

        # Get train patient labels
        train_patients = set([])
        for fid in train_files:
            patient = self.get_patient_label(fid)
            train_patients = train_patients.union(set([patient]))

        # Sort by age, then image_label
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

    def make_train_test_split(self):
        # Train test split done above
        pass


class MRISamePatientPairStream(MRISingleStream):
    def group_data(self):
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

    def make_train_test_split(self):
        # split is done during sampling
        pass


class MRIDifferentPatientPairStream(MRISingleStream):
    def group_data(self):
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

    def make_train_test_split(self):
        # split is done during sampling
        pass


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

    def group_data(self):
        self.start_time = process_time()
        n_pairs = self.config["n_pairs"]
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
        # collect patient labels
        patient_labels = set([])
        for d in diagnoses_to_sample:
            labels = set(diag_to_patient[d].keys())
            patient_labels = patient_labels.union(labels)

        # prepare train-test split
        patient_labels = sorted(list(patient_labels))
        n_patients = len(patient_labels)
        n_train = int(self.config["train_ratio"] * n_patients)
        n_train_pairs = int(self.config["train_ratio"] * n_pairs)
        train_labels = patient_labels[:n_train]
        test_labels = patient_labels[n_train:]

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
            warnings.warn("Sampled only {} pairs!!!".format(len(sampled)))

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

            assert (self.get_patient_id(id1) == self.get_patient_id(id2)) == \
                self.config["same_patient"]

        return groups

    def make_train_test_split(self):
        # split is done during sampling
        pass


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

    def group_data(self):
        max_train_pairs = self.get_max_train_pairs()
        diagnoses = self.get_diagnoses()

        # Use train-test split of MRISingleStream. Makes balanced split
        # with respect to diagnoses. In addition, the train and test set
        # of patients is distinct.
        self.groups = self.make_one_sample_groups()
        self.make_balanced_train_test_split()

        # Collect train and test patients
        train_files = [g.file_ids[0] for g in self.groups
                       if g.is_train == True]
        test_files = [g.file_ids[0] for g in self.groups
                      if g.is_train == False]

        self.groups = None
        # Make arbitrary test 
        groups = []
        test_groups = self.produce_test_groups(test_files, 2)
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

    def make_train_test_split(self):
        pass
