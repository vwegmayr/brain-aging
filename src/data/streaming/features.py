import tensorflow as tf


class FeatureCollection:
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


adni_aibl = FeatureCollection()

IMAGE_LABEL = adni_aibl.add(
    'image_label', t=tf.string, default='')
SUBJECT_LABEL = adni_aibl.add(
    'patient_label', t=tf.string, default='')
DATASET = adni_aibl.add(
    'dataset', t=tf.string, default='')


for disease in [
    'ad', 'lmci', 'emci', 'mci', 'pd', 'smc',
    'prodromal', 'swedd', 'gencohort_unaff', 'gencohort_pd',
]:
    adni_aibl.add('health_%s' % disease, disease, default=0)


HEALTHY = adni_aibl.add('healthy', 'hc', default=0)
MRI = adni_aibl.add('mri', t=tf.float32)
SEX = adni_aibl.add('sex', doc='male = 0; female = 1', default=-1)
STUDY_ID = adni_aibl.add('study_id', 'study', default=-1)
STUDY_IMAGE_ID = adni_aibl.add('study_image_id', 'image', default=-1)
STUDY_PATIENT_ID = adni_aibl.add('study_patient_id', 'patient', default=-1)

collections = {
    "adni_aibl": adni_aibl
}
