import tensorflow as tf


class FeatureCollection:
    def __init__(self):
        self.feature_info = {}

    def add(
        self,
        s,
        shortname=None,
        t=tf.int64,
        py_type=int,
        shape=[1],
        doc=None,
        default=None,
    ):
        self.feature_info[s] = {
            'type': t,
            'shortname': shortname if shortname is not None else s,
            'shape': shape,
            'default': default,
            'py_type': py_type,
        }
        return s


adni_aibl = FeatureCollection()

IMAGE_LABEL = adni_aibl.add(
    'image_label', t=tf.string, py_type=str, default='')
SUBJECT_LABEL = adni_aibl.add(
    'patient_label', t=tf.string, py_type=str, default='')
#DATASET = adni_aibl.add(
 #   'dataset', t=tf.string, py_type=str, default='')


for disease in [
    'lmci', 'emci', 'mci', 'smc', 'ad'
    #'prodromal', 'swedd', 'gencohort_unaff', 'gencohort_pd',
]:
    adni_aibl.add('health_%s' % disease, disease, default=0)

AGE = adni_aibl.add("age", default=-1)
HEALTHY = adni_aibl.add('healthy', 'hc', default=0)
MRI = adni_aibl.add('mri', t=tf.float32, py_type=float)
SEX = adni_aibl.add('sex', doc='male = 0; female = 1', default=-1)
STUDY_ID = adni_aibl.add('study_id', 'study', default=-1)
STUDY_IMAGE_ID = adni_aibl.add('study_image_id', 'image', default=-1)
STUDY_PATIENT_ID = adni_aibl.add('study_patient_id', 'patient', default=-1)

collections = {
    "adni_aibl": adni_aibl
}
