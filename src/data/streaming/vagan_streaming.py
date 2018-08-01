import numpy as np
import copy

from src.data.streaming.mri_streaming import MRISingleStream
from src.baum_vagan.utils import map_image_to_intensity_range


class BatchProvider(object):
    def __init__(self, streamer, file_ids, label_key, prefetch=1000):
        self.file_ids = file_ids
        assert len(file_ids) > 0
        self.streamer = streamer
        self.label_key = label_key
        self.prefetch = prefetch
        self.loaded = []
        self.np_random = np.random.RandomState(seed=11)
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
    def __init__(self, streamer, samples, label_key, prefetch=100):
        self.samples = samples
        assert len(samples) > 0
        self.streamer = streamer
        self.label_key = label_key
        self.prefetch = prefetch
        self.loaded = []
        self.np_random = np.random.RandomState(seed=11)
        self.sample_gen = self.next_sample()
        self.img_gen = self.next_image()

    def next_sample(self):
        self.np_random.shuffle(self.samples)
        p = 0
        while (1):
            if p < len(self.samples):
                yield self.samples[p]
                p += 1
            else:
                p = 0
                self.np_random.shuffle(self.samples)

    def next_image(self):
        loaded = []

        while (1):
            if len(loaded) == 0:
                # prefetch
                for i in range(self.prefetch):
                    sample = next(self.sample_gen)
                    # labels are not used by VAGAN, only needed
                    # for compatibility
                    label = -1
                    im = sample.load()
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
        super(MRIImagePair, self).__init__(
            streamer=streamer
        )
        self.fid1 = fid1
        self.fid2 = fid2

    def get_age_delta(self):
        age2 = self.streamer.get_exact_age(self.fid2)
        age1 = self.streamer.get_exact_age(self.fid1)

        return age2 - age1

    def load_image(self, fid):
        p = self.streamer.get_file_path(fid)
        im = self.streamer.load_sample(p)
        if self.streamer.normalize_images:
            im = self.streamer.normalize_image(im)
        if self.streamer.rescale_to_one:
            im = map_image_to_intensity_range(im, -1, 1, 0.05)

        im = np.reshape(im, tuple(list(im.shape) + [1]))
        return im

    def load(self):
        im1 = self.load_image(self.fid1)
        im2 = self.load_image(self.fid2)
        delta_im = im2 - im1
        im = np.concatenate((im1, delta_im), axis=-1)
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

        super(AgeFixedDeltaStream, self).__init__(
            stream_config=stream_config
        )

        self.prefetch = self.config["prefetch"]
        self.set_up_batches()

    def select_file_ids(self, file_ids):
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
                    if delta > self.delta_max:
                        break

                    if delta < self.delta_min:
                        continue

                    diag_i = self.get_diagnose(i_fid)
                    diag_j = self.get_diagnose(j_fid)

                    if diag_i in self.use_diagnoses and \
                            diag_j in self.use_diagnoses:
                        keep_fids.append(i_fid)
                        keep_fids.append(j_fid)

                        with open("labels_delta_1.csv", "a") as f:
                            f.write("{},{}\n".format(i_fid, j_fid))

        return keep_fids

    def build_pairs(self, fids):
        pairs = []

        age_ascending = sorted(
            fids,
            key=lambda x: self.get_exact_age(x)
        )

        n = len(age_ascending)
        for i in range(n):
            age_i = self.get_exact_age(age_ascending[i])
            for j in range(i + 1, n):
                age_j = self.get_exact_age(age_ascending[j])
                delta = age_j - age_i
                if delta > self.delta_max:
                    break

                if delta < self.delta_min:
                    continue

                pairs.append(MRIImagePair(
                    streamer=self,
                    fid1=age_ascending[i],
                    fid2=age_ascending[j]
                ))

        return pairs

    def set_up_batches(self):
        # Train batches
        train_ids = self.get_train_ids()
        train_pairs = self.build_pairs(train_ids)

        # Some checks
        for p in train_pairs:
            assert p.get_age_delta() >= self.delta_min
            assert p.get_age_delta() <= self.delta_max

        self.trainAD = FlexibleBatchProvider(
            streamer=self,
            samples=train_pairs,
            label_key=None,
            prefetch=self.prefetch
        )
        self.trainCN = FlexibleBatchProvider(
            streamer=self,
            samples=train_pairs,
            label_key=None,
            prefetch=self.prefetch
        )

        # Validation batches
        val_ids = self.get_validation_ids()
        val_pairs = self.build_pairs(val_ids)
        self.validationAD = FlexibleBatchProvider(
            streamer=self,
            samples=val_pairs,
            label_key=None,
            prefetch=self.prefetch
        )
        self.validationCN = FlexibleBatchProvider(
            streamer=self,
            samples=val_pairs,
            label_key=None,
            prefetch=self.prefetch
        )

        # Test batches
        test_ids = self.get_test_ids()
        test_pairs = self.build_pairs(test_ids)
        self.testAD = FlexibleBatchProvider(
            streamer=self,
            samples=test_pairs,
            label_key=None,
            prefetch=self.prefetch
        )
        self.testCN = FlexibleBatchProvider(
            streamer=self,
            samples=test_pairs,
            label_key=None,
            prefetch=self.prefetch
        )
