from .mri_streaming import MRIDiagnosePairStream
from .mri_streaming import MRISamePatientSameAgePairStream


class MRIDiagnosePairStreamCollection(object):
    def __init__(self, stream_config, diagnosis_pairs):
        self.streamers = []
        self.stream_config = stream_config
        self.diag_pair_to_streamers = {}
        self.diagnosis_pairs = diagnosis_pairs
        self.pair_to_same_patient = {}
        self.pair_to_different_patient = {}
        # Create Streamers
        for p in diagnosis_pairs["same_patient"]:
            p = tuple(p)
            self.diag_pair_to_streamers[p] = []
            stream_config['diagnoses'] = p
            # Same patient
            stream_config['same_patient'] = True
            streamer = MRIDiagnosePairStream(stream_config)
            name = self.diagnosis_pair_to_str(p)
            streamer.name = 'different_patient_' + name
            self.streamers.append(streamer)
            self.diag_pair_to_streamers[p].append(streamer)
            self.pair_to_same_patient[p] = streamer
        for p in diagnosis_pairs["different_patient"]:
            p = tuple(p)
            stream_config['diagnoses'] = p
            if p not in self.diag_pair_to_streamers:
                self.diag_pair_to_streamers[p] = []
            # Different patient
            stream_config['same_patient'] = False
            streamer = MRIDiagnosePairStream(stream_config)
            name = self.diagnosis_pair_to_str(p)
            streamer.name = 'same_patient_' + name
            self.streamers.append(streamer)
            self.diag_pair_to_streamers[p].append(streamer)
            self.pair_to_different_patient[p] = streamer

        # Add special streamer
        streamer = MRISamePatientSameAgePairStream(stream_config)
        streamer.name = "same_patient_same_age_same_diag"
        self.streamers.append(streamer)

    def get_same_patient_streamers(self):
        return [self.pair_to_same_patient[tuple(p)]
                for p in self.diagnosis_pairs["same_patient"]]

    def get_different_patient_streamers(self):
        return [self.pair_to_different_patient[tuple(p)]
                for p in self.diagnosis_pairs["different_patient"]]

    def diagnosis_pair_to_str(self, pair):
        return pair[0] + "__" + pair[1]

    def get_diagnosis_pairs(self):
        return list(self.diag_pair_to_streamers.keys())

    def diagnosis_pair_to_streamers(self, pair):
        return self.diag_pair_to_streamers[pair]

    def get_streamers(self):
        return self.streamers
