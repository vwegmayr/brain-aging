import numpy as np
import copy
import pydoc
from collections import OrderedDict
import itertools

from src.data.streaming.mri_streaming import MRISingleStream
from src.baum_vagan.utils import map_image_to_intensity_range
from src.data.streaming.base import Group


class BatchProvider(object):
    def __init__(self, streamer, file_ids, label_key, prefetch=1000, seed=11):
        self.file_ids = file_ids
        assert len(file_ids) > 0
        self.streamer = streamer
        self.label_key = label_key
        self.prefetch = prefetch
        self.loaded = []
        self.np_random = np.random.RandomState(seed=seed)
        self.fid_gen = self.next_fid()
        self.img_gen = self.next_image()

    def next_fid(self):
        self.np_random.shuffle(self.file_ids)
        p = 0
        while (1):
            if p < len(self.file_ids):
                yield self.file_ids[p]
                p += 1
            else:
                p = 0
                self.np_random.shuffle(self.file_ids)

    def next_image(self):
        loaded = []

        while (1):
            if len(loaded) == 0:
                # prefetch
                for i in range(self.prefetch):
                    fid = next(self.fid_gen)
                    p = self.streamer.get_file_path(fid)
                    im = self.streamer.load_sample(p)
                    if self.streamer.normalize_images:
                        im = self.streamer.normalize_image(im)
                    label = self.streamer.get_meta_info_by_key(
                        fid, self.label_key
                    )
                    im = np.reshape(im, tuple(list(im.shape) + [1]))
                    loaded.append([im, label])
            else:
                el = loaded[0]
                loaded.pop(0)
                yield el

    def next_batch(self, batch_size):
        X_batch = []
        y_batch = []
        for i in range(batch_size):
            x, y = next(self.img_gen)
            X_batch.append(x)
            y_batch.append(y)

        return np.array(X_batch), np.array(y_batch)


class FlexibleBatchProvider(object):
    def __init__(self, streamer, samples, label_key, prefetch=100, seed=11):
        self.samples = samples
        self.indices = list(range(len(samples)))
        assert len(samples) > 0
        self.streamer = streamer
        self.label_key = label_key
        self.prefetch = prefetch
        self.loaded = []
        self.np_random = np.random.RandomState(seed=seed)
        self.idx_gen = self.next_idx()
        self.img_gen = self.next_image()

    def next_idx(self):
        self.np_random.shuffle(self.indices)
        p = 0
        while (1):
            if p < len(self.indices):
                yield self.indices[p]
                p += 1
            else:
                p = 0
                self.np_random.shuffle(self.indices)

    def next_image(self):
        loaded = []

        while (1):
            if self.prefetch > 0:
                if len(loaded) == 0:
                    # prefetch
                    for i in range(self.prefetch):
                        idx = next(self.idx_gen)
                        sample = self.samples[idx]
                        # labels are not used by VAGAN, only needed
                        # for compatibility
                        label = (sample.fid1, sample.fid2)
                        im = sample.load()
                        loaded.append([im, label])
                else:
                    el = loaded[0]
                    loaded.pop(0)
                    yield el
            else:
                idx = next(self.idx_gen)
                sample = self.samples[idx]
                # labels are not used by VAGAN, only needed
                # for compatibility
                label = (sample.fid1, sample.fid2)
                im = sample.load()
                yield im, label

    def next_batch(self, batch_size):
        X_batch = []
        y_batch = []
        for i in range(batch_size):
            x, y = next(self.img_gen)
            X_batch.append(x)
            y_batch.append(y)

        return np.array(X_batch), np.array(y_batch)


class SameDeltaBatchProvider(object):
    def __init__(self, streamer, samples, label_key, prefetch=100, seed=11):
        self.samples = samples

        assert len(samples) > 0
        self.streamer = streamer
        self.label_key = label_key
        self.prefetch = prefetch
        self.loaded = []
        self.seed = seed

        self.match_delta_to_samples()

        # one provider for each delta
        self.delta_to_provider = {}
        for delta in self.deltas:
            self.delta_to_provider[delta] = FlexibleBatchProvider(
                streamer=streamer,
                samples=self.delta_to_samples[delta],
                label_key=label_key,
                prefetch=prefetch,
                seed=seed
            )

    def match_delta_to_samples(self):
        self.delta_to_samples = OrderedDict()
        for sample in self.samples:
            approx_delta = sample.get_approx_delta()
            if approx_delta not in self.delta_to_samples:
                self.delta_to_samples[approx_delta] = [sample]
            else:
                self.delta_to_samples[approx_delta].append(sample)

        self.deltas = sorted(self.delta_to_samples.keys())
        self.delta_gen = itertools.cycle(self.deltas)

    def next_batch(self, batch_size):
        delta = next(self.delta_gen)
        provider = self.delta_to_provider[delta]

        X_batch, y_batch = provider.next_batch(batch_size)
        return X_batch, y_batch


class AnySingleStream(MRISingleStream):
    """
    Compatibility with Baumgartner VAGAN implementation.
    """
    def __init__(self, *args, **kwargs):
        super(AnySingleStream, self).__init__(
            *args,
            **kwargs
        )
        self.AD_key = self.config["AD_key"]  
        self.CN_key = self.config["CN_key"]  # Control group
        self.set_up_batches()

    def get_ad_cn_ids(self, file_ids):
        ad_ids = []
        cn_ids = []

        for fid in file_ids:
            v = self.get_meta_info_by_key(fid, self.AD_key)
            if v == 1:
                ad_ids.append(fid)
            else:
                cn_ids.append(fid)

        return ad_ids, cn_ids

    def set_up_batches(self):
        # Train batches
        train_ids = self.get_train_ids()
        train_AD_ids, train_CN_ids = self.get_ad_cn_ids(train_ids)
        self.n_train_samples = len(train_AD_ids)
        self.trainAD = BatchProvider(
            streamer=self,
            file_ids=train_AD_ids,
            label_key=self.AD_key,
            prefetch=self.config["prefetch"]
        )
        self.trainCN = BatchProvider(
            streamer=self,
            file_ids=train_CN_ids,
            label_key=self.CN_key,
            prefetch=self.config["prefetch"]
        )

        # Validation batches
        validation_ids = self.get_validation_ids()
        valid_AD_ids, valid_CN_ids = self.get_ad_cn_ids(validation_ids)
        self.n_val_samples = len(valid_AD_ids)
        self.validationAD = BatchProvider(
            streamer=self,
            file_ids=valid_AD_ids,
            label_key=self.AD_key,
            prefetch=self.config["prefetch"]
        )
        self.validationCN = BatchProvider(
            streamer=self,
            file_ids=valid_CN_ids,
            label_key=self.CN_key,
            prefetch=self.config["prefetch"]
        )

        # Test batches
        test_ids = self.get_test_ids()
        test_AD_ids, test_CN_ids = self.get_ad_cn_ids(test_ids)
        self.n_test_samples = len(test_AD_ids)
        self.testAD = BatchProvider(
            streamer=self,
            file_ids=test_AD_ids,
            label_key=self.AD_key,
            prefetch=self.config["prefetch"]
        )
        self.testCN = BatchProvider(
            streamer=self,
            file_ids=test_CN_ids,
            label_key=self.CN_key,
            prefetch=self.config["prefetch"]
        )


class MRISample(object):
    def __init__(self, streamer):
        self.streamer = streamer

    def load(self):
        pass


class MRIImagePair(MRISample):
    def __init__(self, streamer, fid1, fid2):
        """
        Args:
            - streamer: MRI streamer
            - fid1: file ID of MRI image
            - fid2: file ID of MRI image
        """
        super(MRIImagePair, self).__init__(
            streamer=streamer
        )
        self.fid1 = fid1
        self.fid2 = fid2
        self.age2 = self.streamer.get_exact_age(self.fid2)
        self.age1 = self.streamer.get_exact_age(self.fid1)
        self.delta = self.age2 - self.age1

        self.slice = False
        if self.streamer.load_only_slice():
            self.slice = True
            self.slice_axis, self.slice_idx = self.streamer.get_slice_info()

        self.raw_data = None

    def set_approx_delta(self, delta):
        self.approx_delta = delta

    def get_approx_delta(self):
        return self.approx_delta

    def get_age_delta(self):
        return self.delta

    def get_diagnoses(self):
        return [
            self.streamer.get_diagnose(self.fid1),
            self.streamer.get_diagnose(self.fid2)
        ]

    def same_patient(self):
        return self.streamer.get_patient_id(self.fid1) == \
            self.streamer.get_patient_id(self.fid2)

    def load_image(self, fid):
        p = self.streamer.get_file_path(fid)
        im = self.streamer.load_raw_sample(p)
        if self.streamer.normalize_images:
            im = self.streamer.normalize_image(im)
        if self.streamer.rescale_to_one:
            im = map_image_to_intensity_range(im, -1, 1, 5)
        if self.slice:
            im = np.take(im, self.slice_idx, axis=self.slice_axis)
        im = np.reshape(im, tuple(list(im.shape) + [1]))
        return im

    def load(self):
        if not self.slice or self.raw_data is None:
            im1 = self.load_image(self.fid1)
            im2 = self.load_image(self.fid2)

            delta_im = im2 - im1
            im = np.concatenate((im1, delta_im), axis=-1)

            if self.slice:
                self.raw_data = im

        elif self.slice and self.raw_data is not None:
            im = self.raw_data

        return im


class MRIImagePairWithDelta(MRIImagePair):
    def __init__(self, *args, **kwargs):
        super(MRIImagePairWithDelta, self).__init__(
            *args,
            **kwargs
        )

        approx_delta = -1
        for approx, delta_range in self.streamer.delta_ranges.items():
            if delta_range[0] <= self.get_age_delta() <= delta_range[1]:
                approx_delta = approx
                break

        assert approx_delta >= 0
        self.set_approx_delta(approx_delta)

    def load(self):
        if not self.slice or self.raw_data is None:
            im1 = self.load_image(self.fid1)
            im2 = self.load_image(self.fid2)
            delta_im = im2 - im1
            delta = self.get_approx_delta()
            delta_channel = 0 * im1 + delta
            im = np.concatenate((im1, delta_channel, delta_im), axis=-1)

            if self.slice:
                self.raw_data = im

        elif self.slice and self.raw_data is not None:
            im = self.raw_data

        return im


class AgeFixedDeltaStream(MRISingleStream):
    def __init__(self, stream_config):
        config = copy.deepcopy(stream_config)
        # Age difference tolerance between two images
        self.delta_min = config["delta_min"]
        self.delta_max = config["delta_max"]
        # True if multiple images taken the same day
        # should be used.
        self.use_retest = config["use_retest"]
        # Diagnoses that should be used
        self.use_diagnoses = config["use_diagnoses"]
        # True if patients having multiple distinct diagnoses
        # should be used
        self.use_converting = config["use_converting"]
        # Rescaling
        self.rescale_to_one = config["rescale_to_one"]
        # Different patient order for real and fake samples
        self.shuffle_real_fake = config["shuffle_real_fake"]

        self.provider = FlexibleBatchProvider
        self.image_type = MRIImagePair
        super(AgeFixedDeltaStream, self).__init__(
            stream_config=stream_config
        )

        self.prefetch = self.config["prefetch"]
        self.set_up_batches()

    def is_valid_delta(self, delta):
        return delta >= self.delta_min and delta <= self.delta_max

    def select_file_ids(self, file_ids):
        file_ids = sorted(file_ids)
        patient_groups = self.make_patient_groups(file_ids)
        keep_fids = []
        for g in patient_groups:
            # Analyze patient
            # Determine if patient can be used
            diagnoses = set([])
            for fid in g.file_ids:
                diag = self.get_diagnose(fid)
                diagnoses.add(diag)
            # multiple diagnoses
            if len(diagnoses) > 1 and not self.use_converting:
                continue

            # Determine which images can be used
            # Sort by ascending age
            age_ascending = sorted(
                g.file_ids,
                key=lambda x: self.get_exact_age(x)
            )

            # Filter out same age images
            n = len(age_ascending)
            if not self.use_retest:
                to_remove = []
                for i in range(1, n):
                    prev_age = self.get_exact_age(age_ascending[i - 1])
                    cur_age = self.get_exact_age(age_ascending[i])
                    if prev_age == cur_age:
                        to_remove.append(i)
                to_remove.reverse()
                for i in to_remove:
                    del age_ascending[i]

            n = len(age_ascending)
            # Consider all possible pairs of images
            for i in range(n):
                i_fid = age_ascending[i]
                age_i = self.get_exact_age(i_fid)
                for j in range(i + 1, n):
                    j_fid = age_ascending[j]
                    age_j = self.get_exact_age(j_fid)
                    delta = age_j - age_i
                    if not self.is_valid_delta(delta):
                        continue

                    diag_i = self.get_diagnose(i_fid)
                    diag_j = self.get_diagnose(j_fid)

                    if diag_i in self.use_diagnoses and \
                            diag_j in self.use_diagnoses:
                        keep_fids.append(i_fid)
                        keep_fids.append(j_fid)

        return keep_fids

    def build_pairs(self, fids):
        pairs = []
        fids = self.select_file_ids(fids)
        patient_groups = self.make_patient_groups(fids)
        for g in patient_groups:
            g_ids = g.file_ids
            age_ascending = sorted(
                g_ids,
                key=lambda x: self.get_exact_age(x)
            )

            n = len(age_ascending)
            for i in range(n):
                age_i = self.get_exact_age(age_ascending[i])
                for j in range(i + 1, n):
                    age_j = self.get_exact_age(age_ascending[j])
                    delta = age_j - age_i

                    if not self.is_valid_delta(delta):
                        continue

                    pairs.append(self.image_type(
                        streamer=self,
                        fid1=age_ascending[i],
                        fid2=age_ascending[j]
                    ))

        return pairs

    def check_pairs(self, pairs):
        # Some checks
        for p in pairs:
            assert self.is_valid_delta(p.get_age_delta())
            assert p.same_patient()
            for diag in p.get_diagnoses():
                assert diag in self.use_diagnoses

    def group_data(self, train_ids, test_ids):
        train_pairs = self.build_pairs(train_ids)
        test_pairs = self.build_pairs(test_ids)

        train_groups = [
            Group(file_ids=[p.fid1, p.fid2], is_train=True)
            for p in train_pairs
        ]

        test_groups = [
            Group(file_ids=[p.fid1, p.fid2], is_train=False)
            for p in test_pairs
        ]

        return train_groups + test_groups

    def set_up_batches(self):
        # Train batches
        train_ids = self.get_train_ids()
        train_pairs = self.build_pairs(train_ids)
        self.n_train_samples = len(train_pairs)
        self.check_pairs(train_pairs)

        if self.shuffle_real_fake:
            real_seed = 11
            fake_seed = 43
        else:
            real_seed = 11
            fake_seed = 11

        self.trainAD = self.provider(
            streamer=self,
            samples=train_pairs,
            label_key=None,
            prefetch=self.prefetch,
            seed=fake_seed
        )
        self.trainCN = self.provider(
            streamer=self,
            samples=train_pairs,
            label_key=None,
            prefetch=self.prefetch,
            seed=real_seed
        )
        self.train_pairs = train_pairs

        # Validation batches
        val_ids = self.get_validation_ids()
        val_pairs = self.build_pairs(val_ids)
        self.n_val_samples = len(val_pairs)
        self.check_pairs(val_pairs)
        self.validationAD = self.provider(
            streamer=self,
            samples=val_pairs,
            label_key=None,
            prefetch=self.prefetch
        )
        self.validationCN = self.provider(
            streamer=self,
            samples=val_pairs,
            label_key=None,
            prefetch=self.prefetch
        )
        self.val_pairs = val_pairs

        # Test batches
        test_ids = self.get_test_ids()
        test_pairs = self.build_pairs(test_ids)
        self.n_test_samples = len(test_pairs)
        self.check_pairs(test_pairs)
        self.testAD = self.provider(
            streamer=self,
            samples=test_pairs,
            label_key=None,
            prefetch=10
        )
        self.testCN = self.provider(
            streamer=self,
            samples=test_pairs,
            label_key=None,
            prefetch=10
        )
        self.test_pairs = test_pairs

    def get_all_pairs(self):
        return self.train_pairs + self.val_pairs + self.test_pairs


class Patient(object):
    def __init__(self, patient_id, delta_to_pairs):
        self.patient_id = patient_id
        self.delta_to_pairs = delta_to_pairs


class PairCollection(object):
    def __init__(self, pairs):
        self.delta_to_pairs = self.compute_delta_to_pairs(pairs)

    def get_pairs(self):
        all_pairs = []
        for pairs in self.delta_to_pairs.values():
            all_pairs += pairs

        return all_pairs

    def get_fids(self):
        fids = set()
        for p in self.get_pairs():
            fids.add(p.fid1)
            fids.add(p.fid2)

        return sorted(list(fids))

    def get_patients(self):
        patient_ids = set()
        for p in self.get_pairs():
            pid = p.streamer.get_patient_id(p.fid1)
            patient_ids.add(pid)

        return sorted(list(patient_ids))

    def compute_delta_to_pairs(self, pairs):
        delta_to_pairs = OrderedDict()
        for pair in pairs:
            delta = pair.get_approx_delta()
            if delta not in delta_to_pairs:
                delta_to_pairs[delta] = set([pair])
            else:
                delta_to_pairs[delta].add(pair)

        return delta_to_pairs

    def get_patient_pairs(self, patient_id):
        patient_pairs = []
        for delta, pairs in self.delta_to_pairs.items():
            for p in pairs:
                fid = p.fid1
                pid = p.streamer.get_patient_id(fid)
                if pid == patient_id:
                    patient_pairs.append(p)

        return patient_pairs

    def add_pairs(self, pairs):
        for pair in pairs:
            delta = pair.get_approx_delta()
            if delta not in self.delta_to_pairs:
                self.delta_to_pairs[delta] = set([pair])
            else:
                self.delta_to_pairs[delta].add(pair)

    def remove_pairs(self, pairs):
        B = set(pairs)
        for delta in self.delta_to_pairs.keys():
            self.delta_to_pairs[delta] -= B

    def compute_delta_score(self, add_patient, remove_patient, delta_targets):
        init_score = 0
        new_score = 0
        for delta, pairs in self.delta_to_pairs.items():
            init_score += abs(delta_targets[delta] - len(pairs))
            diff = 0
            for p in pairs:
                fid = p.fid1
                patient_id = p.streamer.get_patient_id(fid)
                if patient_id == add_patient:
                    diff += 1
                if patient_id == remove_patient:
                    diff -= 1

            new_score += abs(delta_targets[delta] - (len(pairs) + diff))

        return new_score - init_score

    def print_counts(self, targets):
        for delta, pairs in self.delta_to_pairs.items():
            print("Delta {}, got {}, target {}".format(
                delta, len(pairs), targets[delta]
            ))


class AgeVariableDeltaStream(AgeFixedDeltaStream):
    def __init__(self, stream_config):
        config = copy.deepcopy(stream_config)
        # Dictionary mapping deltas to ranges
        # Make sure keys are floats
        self.delta_ranges = OrderedDict()
        for k, v in config["delta_ranges"].items():
            self.delta_ranges[float(k)] = v

        # True if multiple images taken the same day
        # should be used.
        self.use_retest = config["use_retest"]
        # Diagnoses that should be used
        self.use_diagnoses = config["use_diagnoses"]
        # True if patients having multiple distinct diagnoses
        # should be used
        self.use_converting = config["use_converting"]
        # Rescaling
        self.rescale_to_one = config["rescale_to_one"]

        # Batch provider class
        self.provider = pydoc.locate(config["batch_provider"])
        self.image_type = MRIImagePairWithDelta

        # Different patient order for real and fake samples
        self.shuffle_real_fake = config["shuffle_real_fake"]

        super(AgeFixedDeltaStream, self).__init__(
            stream_config=stream_config
        )

        # How many images are loaded into memory. If equal
        # to -1, all images are loaded.
        self.prefetch = self.config["prefetch"]
        self.set_up_batches()
        if not self.silent:
            self.print_some_stats()

    def get_delta_to_pairs(self, pairs):
        """
        Arg:
            - pairs: a list of image pairs
        Return:
            - dictionary mapping deltas to
              image pairs
        """
        delta_to_pairs = OrderedDict()
        for pair in pairs:
            delta = pair.get_approx_delta()
            if delta not in delta_to_pairs:
                delta_to_pairs[delta] = [pair]
            else:
                delta_to_pairs[delta].append(pair)

        return delta_to_pairs

    def _rebalance(self, train_ids, test_ids):
        """
        Poor attempt to rebalance samples across
        train-val-test split with respect to deltas.
        Pretty slow, does not work really well.
        Don't use it.
        """
        self.np_random.shuffle(train_ids)
        self.np_random.shuffle(test_ids)
        train_pairs = self.build_pairs(train_ids)
        test_pairs = self.build_pairs(test_ids)

        delta_targets = self.get_delta_to_pairs(train_pairs + test_pairs)
        train_targets = {}
        test_targets = {}
        for delta, pairs in delta_targets.items():
            test_target = len(pairs) * 1.0 / self.config["n_folds"]
            test_targets[delta] = int(test_target)
            train_targets[delta] = len(pairs) - int(test_target)

        train_collection = PairCollection(train_pairs)
        test_collection = PairCollection(test_pairs)

        print(">>>>>>>> before")
        train_collection.print_counts(train_targets)
        test_collection.print_counts(test_targets)

        train_patients = train_collection.get_patients()
        test_patients = test_collection.get_patients()

        for train_p in train_patients:
            best_score = 0
            best_p = None
            for test_p in test_patients:
                delta_sc = 0
                sc_0 = train_collection.compute_delta_score(
                    add_patient=test_p,
                    remove_patient=train_p,
                    delta_targets=train_targets
                )

                sc_1 = test_collection.compute_delta_score(
                    add_patient=train_p,
                    remove_patient=test_p,
                    delta_targets=test_targets
                )
                delta_sc = sc_0 + sc_1
                if sc_0 < 0 and sc_1 < 0 and delta_sc < best_score:
                    best_score = delta_sc
                    best_p = test_p

            if best_score < 0:
                new_train_pairs = test_collection.get_patient_pairs(best_p)
                train_collection.add_pairs(new_train_pairs)

                new_test_pairs = train_collection.get_patient_pairs(train_p)
                test_collection.add_pairs(new_test_pairs)

                train_collection.remove_pairs(new_test_pairs)
                test_collection.remove_pairs(new_train_pairs)

                # train_collection.print_counts(train_targets)

        new_train_ids = train_collection.get_fids()
        new_test_ids = test_collection.get_fids()

        print(">>>>>>>> after")
        train_collection.print_counts(train_targets)
        test_collection.print_counts(test_targets)

        return new_train_ids, new_test_ids

    """
    def rebalance_ids(self, train_ids, validation_ids, test_ids):
        train_ids, validation_ids = self._rebalance(train_ids, validation_ids)
        train_ids, test_ids = self._rebalance(train_ids, test_ids)

        return train_ids, validation_ids, test_ids
    """

    def print_some_stats(self):
        def print_provider(provider):
            for delta, p in provider.delta_to_provider.items():
                print("Delta {}, nbr samples {}".format(
                    delta, len(p.samples)
                ))

        print("Train samples")
        print_provider(self.trainAD)
        print("Validation samples")
        print_provider(self.validationAD)
        print("Test samples")
        print_provider(self.testAD)

    def is_valid_delta(self, delta):
        for delta_range in self.delta_ranges.values():
            if delta_range[0] <= delta <= delta_range[1]:
                return True

        return False
