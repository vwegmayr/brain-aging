import nibabel as nib
import warnings
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

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
            return [[group] for group in groups]

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
            im = im / np.max(im)
            shape = tuple(self.config["downsample"]["shape"])
            im = resize(im, shape, anti_aliasing=False)

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
                    g = Group([id_1, id_2])
                    g.patient_label = patient_label
                    groups.append(g)

        return groups
