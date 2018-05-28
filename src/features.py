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
    ):
        self.feature_info[s] = {
            'type': t,
            'shortname': shortname if shortname is not None else s,
            'shape': shape,
            'default': default,
        }
        return s


all_features = FeaturesMgr()

# Internal features for extraction
IMAGE_LABEL = all_features.add(
    'image_label', t=tf.string, default='')
SUBJECT_LABEL = all_features.add(
    'patient_label', t=tf.string, default='')
DATASET = all_features.add(
    'dataset', t=tf.string, default='')

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

# Other features
STUDY_ID = all_features.add('study_id', 'study', default=-1)
STUDY_IMAGE_ID = all_features.add('study_image_id', 'image', default=-1)
STUDY_PATIENT_ID = all_features.add('study_patient_id', 'patient', default=-1)
SUBJECT_DIVERSITY = all_features.add(
    'subject_diversity', default=1)

MRI_MANUFACTURER = all_features.add(
    'mri_manufacturer', t=tf.string, default='',
    doc='Manufacturer name e.g. SIEMENS',
)
MRI_FIELD_STRENGTH = all_features.add(
    'mri_field_strength', t=tf.string, default='',
    doc='Field strength in Tesla - if available',
)
