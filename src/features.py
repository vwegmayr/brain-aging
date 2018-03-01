import tensorflow as tf


class FeaturesMgr:
    def __init__(self):
        self.feature_info = {}

    def add(self, s, shortname=None, t=tf.int64, shape=[1], doc=None):
        self.feature_info[s] = {
            'type': t,
            'shortname': shortname if shortname is not None else s,
            'shape': shape,
        }
        return s


all_features = FeaturesMgr()

# List features here
AGE = all_features.add('age')
HEALTH_AD = all_features.add('health_ad', 'ad')
HEALTH_LMCI = all_features.add('health_lmci', 'lmci')
HEALTH_EMCI = all_features.add('health_emci', 'emci')
HEALTH_MCI = all_features.add('health_mci', 'mci')
HEALTH_PD = all_features.add('health_pd', 'pd')
HEALTH_SMC = all_features.add('health_smc', 'smc')
HEALTHY = all_features.add('healthy', 'hc')
MRI = all_features.add('mri', t=tf.float32)
SEX = all_features.add('sex', doc='male = 0; female = 1')
STUDY_ID = all_features.add('study_id', 'study')
STUDY_IMAGE_ID = all_features.add('study_image_id', 'image')
STUDY_PATIENT_ID = all_features.add('study_patient_id', 'patient')
