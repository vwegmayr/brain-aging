import nibabel as nib

from .base import FileStream
from .base import Group


class MRIImageLoader(object):
    def load_image(self, image_path):
        mri_image = nib.load(image_path)
        mri_image = mri_image.get_data()

        return mri_image


class MRISingleStream(FileStream, MRIImageLoader):
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