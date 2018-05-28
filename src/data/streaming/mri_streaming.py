import nibabel as nib
import warnings

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
    def get_batches(self):
        return [[group] for group in self.groups]

    def group_data(self):
        # We just to stream the images one by one
        groups = []
        for key in self.file_id_to_meta:
            if "file_path" in self.file_id_to_meta[key]:
                groups.append(Group([key]))

        return groups

    def load_sample(self, file_path):
        return self.load_image(file_path)

    def make_train_test_split(self):
        pass


class MRISamePatientSameAgePairStream(FileStream):
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
                    record[self.meta_id_column]
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
                    groups.append(Group([id_1, id_2]))

        return groups

    def get_batches(self):
        if self.batch_size == -1:
            return [self.groups]
        else:
            raise NotImplementedError()

    def load_sample(self, file_path):
        return self.load_image(file_path)

    def make_train_test_split(self):
        pass
