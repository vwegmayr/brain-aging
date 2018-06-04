from .mri_streaming import MRIDiagnosePairStream
from .mri_streaming import MRISamePatientSameAgePairStream


class MRIDiagnosePairStreamCollection(object):
    def __init__(self, stream_config, diagnosis_pairs):
        self.streamers = []
        self.stream_config = stream_config
        # Create Streamers
        for p in diagnosis_pairs["same_patient"]:
            stream_config['diagnoses'] = p
            # Same patient
            stream_config['same_patient'] = True
            streamer = MRIDiagnosePairStream(stream_config)
            name = self.diagnosis_pair_to_str(p)
            streamer.name = 'different_patient_' + name
            self.streamers.append(streamer)
        for p in diagnosis_pairs["different_patient"]:
            stream_config['diagnoses'] = p
            # Different patient
            stream_config['same_patient'] = False
            streamer = MRIDiagnosePairStream(stream_config)
            name = self.diagnosis_pair_to_str(p)
            streamer.name = 'same_patient_' + name
            self.streamers.append(streamer)

        # Add special streamer
        streamer = MRISamePatientSameAgePairStream(stream_config)
        streamer.name = "same_patient_same_age_same_diag"
        self.streamers.append(streamer)

    def diagnosis_pair_to_str(self, pair):
        return pair[0] + "__" + pair[1]

    def get_streamers(self):
        return self.streamers
