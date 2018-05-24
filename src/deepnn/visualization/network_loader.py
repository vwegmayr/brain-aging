import os
import json
import glob
import collections
import tensorflow as tf
from src.data.features_store import FeaturesStore


class NetworkLoader:
    def __init__(self, export_dir):
        meta_file = tf.train.latest_checkpoint(export_dir) + '.meta'

        with open(os.path.join(export_dir, 'dataset.json'), 'r') as fp:
            self.dataset = json.load(fp)
        print('Dataset loaded')

        tf.reset_default_graph()
        self.sess = tf.Session()
        print('Session created')

        new_saver = tf.train.import_meta_graph(meta_file)
        print('Meta graph imported')

        new_saver.restore(self.sess, tf.train.latest_checkpoint(export_dir))
        print('Graph restored')

    def add_sai_smoothed_mci_to_test_set(self):
        health_mci_files = []
        glob_pattern = \
            '/local/ADNI_AIBL/ADNI_AIBL_T1_smoothed/all_images/*_*.nii.gz'
        fs = FeaturesStore(
            csv_file_path='data/raw/csv/adni_aibl.csv',
            features_from_filename={
                'regexp': '.*/([A0-9]+)_(aug_){0,0}mni_aligned\\.nii\\.gz',
                'features_group': {
                      'image_label': 1,
                }
            }
        )

        for file_name in glob.glob(glob_pattern):
            try:
                features = fs.get_features_for_file(file_name)
                if features['health_mci']:
                    health_mci_files.append(file_name)
            except LookupError:
                pass
        self.dataset['test']['health_mci'] = health_mci_files

    def create_test_retest_dataset(
        self,
        classes,
        from_dataset_name='test',
        max_images_per_patient=10,
        max_patients_per_class=5,
    ):
        return_dataset = {}
        file_to_features = self.dataset['file_to_features']

        for class_name in classes:
            class_images = self.dataset[from_dataset_name][class_name]
            patients_ids = [
                file_to_features[fname]['study_patient_id']
                for fname in class_images
            ]
            counter = collections.Counter(patients_ids)
            return_dataset.update({
                '%s__c%s' % (class_name, mc_rank): [
                    fname
                    for fname in class_images
                    if (file_to_features[fname]['study_patient_id'] == p[0]
                        and '_aug' not in fname)  # Skip augmented images
                ][:max_images_per_patient]
                for mc_rank, p in enumerate(counter.most_common(
                    max_patients_per_class))
            })
        return return_dataset

    def create_colorized_dataset(
        self,
        color_fn,
        from_dataset_name='test',
        classes=['healthy', 'health_ad'],
        max_images_per_age=5,
    ):
        return_dataset = {}
        file_to_features = self.dataset['file_to_features']

        for class_name in classes:
            class_images = self.dataset[from_dataset_name][class_name]
            for fname in class_images:
                color_idx = color_fn(file_to_features[fname])
                if color_idx is None:
                    continue
                class_full_name = '%s__c%s' % (class_name, int(color_idx))
                if class_full_name not in return_dataset:
                    return_dataset[class_full_name] = []
                return_dataset[class_full_name].append(fname)
        return {
            k: v[:max_images_per_age]
            for k, v in return_dataset.items()
        }
