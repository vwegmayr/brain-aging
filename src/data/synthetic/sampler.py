import numpy as np
from skimage import filters
import os
import h5py


def four_disks(effect_size=50., image_size=100, moving_effect=True,
          big_rad=15, small_rad=9, big=[True, True, True, True]):
    stdbckg = 50.  # std deviation of the background
    stdkernel = 2.5  # std deviation of the Gaussian smoothing kernel
    img = np.zeros([image_size, image_size])

    def draw_disk(center, radius):
        i_pos = range(center[0] - radius - 1, center[0] + radius + 1)
        j_pos = range(center[1] - radius - 1, center[1] + radius + 1)

        for i in i_pos:
            for j in j_pos:
                # check if inside circle
                if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2:
                    img[i, j] = effect_size

    # Compute disk centers
    offset = 10
    # Large disks
    c1 = (offset + big_rad, offset + big_rad)
    c2 = (offset + big_rad, image_size - offset - big_rad)
    # Small disks
    c3 = (image_size - offset - big_rad, offset + big_rad)
    c4 = (image_size - offset - big_rad, image_size - offset - big_rad)

    centers = [c1, c2, c3, c4]
    for b, c in zip(big, centers):
        if b:
            draw_disk(c, big_rad)
        else:
            draw_disk(c, small_rad)

    noise = np.random.normal(
        scale=stdbckg, size=np.asarray([image_size, image_size])
    )
    smnoise = filters.gaussian(noise, stdkernel)
    smnoise = smnoise / np.std(smnoise) * stdbckg
    img_with_noise = img + smnoise

    return img, img_with_noise


def tzero_not_fixed_delta_fixed(np_random, image_size, effect_size=100, delta=5,
                                big_rads=[10, 12, 15, 17, 20]):
    n = len(big_rads)
    idx = np_random.randint(0, n)
    big_rad = big_rads[idx]
    small_rad = big_rad - delta

    stdbckg = 50.  # std deviation of the background
    stdkernel = 2.5  # std deviation of the Gaussian smoothing kernel
    noise = np_random.normal(
        scale=stdbckg, size=np.asarray([image_size, image_size])
    )
    smnoise = filters.gaussian(noise, stdkernel)
    smnoise = smnoise / np.std(smnoise) * stdbckg

    img_t0, _ = four_disks(
        image_size=image_size,
        effect_size=effect_size,
        small_rad=small_rad,
        big_rad=big_rad,
        big=[True, True, True, True]
    )

    img_t1, _ = four_disks(
        image_size=image_size,
        effect_size=effect_size,
        small_rad=small_rad,
        big_rad=big_rad,
        big=[False, False, False, False]
    )

    return img_t0, img_t1, smnoise


class Sampler(object):
    def __init__(self, n_images, cn_func, cn_params, ad_func, ad_params,
                 outdir):
        self.n_images = n_images
        self.cn_func = cn_func
        self.ad_func = ad_func
        self.cn_params = cn_params
        self.ad_params = ad_params
        self.outdir = outdir

        if not os.path.exists(outdir):
            os.makedirs(outdir)

    def transform(self, X=None, y=None):
        # sample images
        images = []
        labels = []
        gts = []

        n_cn = self.n_images // 2
        n_ad = self.n_images // 2

        for i in range(n_cn):
            labels.append(0)
            gt, img = self.cn_func(**self.cn_params)
            gts.append(gt)
            images.append(img)

        for i in range(n_ad):
            labels.append(1)
            gt, img = self.ad_func(**self.ad_params)
            gts.append(gt)
            images.append(img)

        images = np.array(images)
        labels = np.array(labels)
        gts = np.array(gts)

        # Save images
        out_path = os.path.join(self.outdir, 'samples.hdf5')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('images', data=images, dtype=np.float32)
            f.create_dataset('labels', data=labels, dtype=np.uint8)
            f.create_dataset('gts', data=gts, dtype=np.uint8)


class TZeroNotFixedSampler(object):
    def __init__(self, n_images, sample_params,
                 outdir):
        self.n_images = n_images
        self.sample_params = sample_params
        self.outdir = outdir
        self.sample_params["np_random"] = np.random.RandomState(seed=40)

        if not os.path.exists(outdir):
            os.makedirs(outdir)

    def transform(self, X=None, y=None):
        # sample images
        images = []
        labels = []
        gts = []

        n_cn = self.n_images // 2
        n_ad = self.n_images // 2

        for i in range(n_cn):
            labels.append(0)
            gt_t0, gt_t1, noise = tzero_not_fixed_delta_fixed(**self.sample_params)
            delta_im = gt_t1 - gt_t0
            gts.append(gt_t1)
            images.append(np.stack((gt_t1 + noise, delta_im), axis=-1))

        for i in range(n_ad):
            labels.append(0)
            gt_t0, gt_t1, noise = tzero_not_fixed_delta_fixed(**self.sample_params)
            delta_im = gt_t1 - gt_t0
            gts.append(gt_t1)
            images.append(np.stack((gt_t0 + noise, delta_im), axis=-1))

        images = np.array(images)
        labels = np.array(labels)
        gts = np.array(gts)

        # Save images
        out_path = os.path.join(self.outdir, 'samples.hdf5')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('images', data=images, dtype=np.float32)
            f.create_dataset('labels', data=labels, dtype=np.uint8)
            f.create_dataset('gts', data=gts, dtype=np.uint8)