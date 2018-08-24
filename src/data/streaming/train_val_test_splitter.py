from sklearn.model_selection import train_test_split
import os

from .mri_streaming import MRISingleStream


class MRIDatasetSplitter(MRISingleStream):
    def __init__(self, stream_config):
        super(MRIDatasetSplitter, self).__init__(
            stream_config=stream_config
        )

    def fit(self, X, y):
        return self

    def get_artifical_label(self, patient_id, file_ids):
        pass

    def get_train_ratio(self):
        return self.config["train_ratio"]

    def get_output_folder(self):
        return self.config["split_folder"]

    def dump_train_val_test_split(self, outfolder, train_ids, val_ids, test_ids):
        def dump(fname, fids):
            with open(os.path.join(outfolder, fname), 'w') as f:
                for fid in fids:
                    f.write("{}\n".format(fid))

        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

        dump("train.txt", train_ids)
        dump("validation.txt", val_ids)
        dump("test.txt", test_ids)

    def transform(self, X, y=None):
        # Signature is just for Compatibility purposes
        patient_groups = self.make_patient_groups(self.all_file_ids)

        pids = []
        labels = []
        pid_to_fids = {}
        label_counts = {}
        for g in patient_groups:
            fids = g.file_ids
            patient_id = self.get_patient_id(fids[0])
            pid_to_fids[patient_id] = fids
            label = self.get_artifical_label(patient_id, fids)
            if label is not None:
                pids.append(patient_id)
                labels.append(label)
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

        for k, v in label_counts.items():
            print("{}: {}".format(k, v))

        # Make train-test split
        assert len(pids) == len(labels)
        train_size = self.get_train_ratio()
        train_val_pids, test_pids, train_val_labels, test_labels = train_test_split(
            pids,
            labels,
            train_size=train_size,
            test_size=1 - train_size,
            random_state=self.seed,
            shuffle=True,
            stratify=labels
        )

        # Make train-val split
        train_pids, val_pids, train_labels, val_labels = train_test_split(
            train_val_pids,
            train_val_labels,
            train_size=train_size,
            test_size=1 - train_size,
            random_state=self.seed,
            shuffle=True,
            stratify=train_val_labels
        )

        train_ids = [fid for pid in train_pids for fid in pid_to_fids[pid]]
        val_ids = [fid for pid in val_pids for fid in pid_to_fids[pid]]
        test_ids = [fid for pid in test_pids for fid in pid_to_fids[pid]]

        # Dump everything to files
        self.dump_train_val_test_split(
            outfolder=self.get_output_folder(),
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids
        )


class ConversionSplitter(MRIDatasetSplitter):
    def get_min_max_age_gap(self):
        return self.config["min_max_age_gap"]

    def get_age_ranges(self):
        return self.config["age_ranges"]

    def age_to_range(self, age):
        ranges = self.get_age_ranges()
        for i, r in enumerate(ranges):
            if r[0] <= age <= r[1]:
                return i

        raise ValueError("No range found for given age")

    def get_artifical_label(self, patient_id, file_ids):
        min_max_age_gap = self.get_min_max_age_gap()

        # Sort file IDs by age
        ages = [self.get_exact_age(fid) for fid in file_ids]
        ages = sorted(ages)
        assert all(ages[i] <= ages[i + 1] for i in range(len(ages) - 1))
        max_age_gap = ages[-1] - ages[0]

        if max_age_gap < min_max_age_gap:
            return None

        n_changes = 0
        for i in range(1, len(file_ids)):
            prev_d = self.get_diagnose(file_ids[i - 1])
            cur_d = self.get_diagnose(file_ids[i])

            if cur_d is not prev_d:
                n_changes += 1

        # Check number of conversions
        if n_changes > 1:
            return None

        conv = self.get_diagnose(file_ids[0]) + "/" \
            + self.get_diagnose(file_ids[-1])

        range_id = self.age_to_range(ages[0])

        return str(range_id) + "/" + conv
