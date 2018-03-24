import tensorflow as tf


class FeaturesMgr:
    def __init__(self):
        self.feature_info = {}

    def add(
        self,
        s,
        shortname=None,
        t=tf.int64,
        shape=[1],
        doc=None,
        default=None,
        only_for_extractor=False,
    ):
        self.feature_info[s] = {
            'type': t,
            'shortname': shortname if shortname is not None else s,
            'shape': shape,
            'default': default,
            'only_for_extractor': only_for_extractor,
        }
        return s


all_features = FeaturesMgr()

# Internal features for extraction
IMAGE_LABEL = all_features.add(
    'image_label', 'label', t=tf.string, default='', only_for_extractor=True)
DATASET = all_features.add(
    'dataset', t=tf.string, default='', only_for_extractor=True)

# List features here
AGE = all_features.add('age', default=-1)

for disease in [
    'ad', 'lmci', 'emci', 'mci', 'pd', 'smc',
    'prodromal', 'swedd', 'gencohort_unaff', 'gencohort_pd',
]:
    all_features.add('health_%s' % disease, disease, default=0)

HEALTHY = all_features.add('healthy', 'hc', default=0)
MRI = all_features.add('mri', t=tf.float32)
SEX = all_features.add('sex', doc='male = 0; female = 1', default=-1)
STUDY_ID = all_features.add('study_id', 'study', default=-1)
STUDY_IMAGE_ID = all_features.add('study_image_id', 'image', default=-1)
STUDY_PATIENT_ID = all_features.add('study_patient_id', 'patient', default=-1)
