import tensorflow as tf
import numpy as np


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
FILE_NAME = adni_aibl.add(
    'file_name', t=tf.string, py_type=str, default=''
)
#DATASET = adni_aibl.add(
 #   'dataset', t=tf.string, py_type=str, default='')


for disease in [
    'lmci', 'emci', 'mci', 'smc', 'ad'
    #'prodromal', 'swedd', 'gencohort_unaff', 'gencohort_pd',
]:
    adni_aibl.add('health_%s' % disease, disease, default=0)

AGE = adni_aibl.add("age", default=-1, py_type=np.float32, t=tf.float32)
AGE_EXACT = adni_aibl.add("age_exact", default=-1, py_type=np.float32, t=tf.float32)
HEALTHY = adni_aibl.add('healthy', 'hc', default=0)
MRI = adni_aibl.add('mri', t=tf.float32, py_type=float)
SEX = adni_aibl.add('sex', doc='male = 0; female = 1', default=-1)
STUDY_ID = adni_aibl.add('study_id', 'study', default=-1)
STUDY_IMAGE_ID = adni_aibl.add('study_image_id', 'image', default=-1)
STUDY_PATIENT_ID = adni_aibl.add('study_patient_id', 'patient', default=-1)

CONV_MCI_AD_DELTA_2 = adni_aibl.add("mci_ad_conv_delta_2", default=-1)

MRI_MANUFACTURER = adni_aibl.add(
    'mri_manufacturer', t=tf.string, default='', py_type=str,
    doc='Manufacturer name e.g. SIEMENS',
)
MRI_FIELD_STRENGTH = adni_aibl.add(
    'mri_field_strength', t=tf.string, default='', py_type=str,
    doc='Field strength in Tesla - if available',
)

WEIGHTING = adni_aibl.add(
    'weighting', t=tf.string, default='', py_type=str,
    doc='e.g., T1, T2',
)


collections = {
    "adni_aibl": adni_aibl
}
