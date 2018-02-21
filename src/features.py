import tensorflow as tf


class FeaturesMgr:
    def __init__(self):
        self.feature_info = {}

    def add(self, s, t=tf.int64, shape=[1], doc=None):
        self.feature_info[s] = {
            'type': t,
            'shape': shape,
        }
        return s


all_features = FeaturesMgr()

# List features here
AGE = all_features.add('age')
HEALTH_AD = all_features.add('health_ad')
HEALTH_LMCI = all_features.add('health_lmci')
HEALTH_EMCI = all_features.add('health_emci')
HEALTH_MCI = all_features.add('health_mci')
HEALTH_PD = all_features.add('health_pd')
HEALTH_SMC = all_features.add('health_smc')
HEALTHY = all_features.add('healthy')
MRI = all_features.add('mri', t=tf.float32)
SEX = all_features.add('sex', doc='male = 0; female = 1')
STUDY_ID = all_features.add('study_id')
STUDY_IMAGE_ID = all_features.add('study_image_id')
STUDY_PATIENT_ID = all_features.add('study_patient_id')
