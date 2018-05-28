import numpy as np
import nibabel as nib
import sklearn.decomposition
from sklearn.exceptions import NotFittedError
import pylab
from matplotlib import pyplot as plt


def load_image_filename(fname):
    import scipy.ndimage
    image = nib.load(fname).get_data()
    #if 'PPMI' in fname:
#        image = scipy.ndimage.filters.gaussian_filter(image, 0.5)
    return np.reshape(image, [1] + list(image.shape))


def cname_extract(s, m='m', default='o'):
    s = s.split('__')
    if len(s) == 1:
        return default
    for _s in s[1:]:
        if _s.startswith(m):
            return _s.split(m)[1]
    return default


def colors_gen(cmap, num_colors):
    cm = pylab.get_cmap(cmap)
    return [cm(1.*i/num_colors) for i in range(num_colors)]


class EmbeddingsVisualization:
    def __init__(self, network_loader):
        self.network_loader = network_loader
        self.sess = network_loader.sess

    # Generate features
    def compute_embeddings(self, dataset, embedding_tensors):
        classes_list = []
        embeddings_list = [[] for t in embedding_tensors]
        for class_name, class_images in dataset.items():
            for fname in class_images:
                sess_run_result = self.sess.run(embedding_tensors, {
                    'is_training:0': False,
                    'input_features/mri:0': load_image_filename(fname),
                })
                assert(len(sess_run_result) == len(embeddings_list))
                for eval_tensor, add_to_list in zip(
                    sess_run_result,
                    embeddings_list,
                ):
                    add_to_list.append(eval_tensor.reshape([-1]))
                classes_list.append(class_name)
        return embeddings_list, classes_list

    def plot_embeddings_pca(
        self,
        dataset,
        legend,
        embedding_tensors=['network_body/conv7/output:0'],
        custom_colors_per_class={},
        decomposition=None,
        plt_legend=True,
        plt_scatter_kwargs={'alpha': 0.6, 's': 20},
    ):
        all_embeddings_list, classes_list = self.compute_embeddings(
            dataset, embedding_tensors)
        full_classes_list_unique = sorted(list(set(classes_list)))
        unique_classes = sorted(list(set([
            c.split('__')[0]
            for c in classes_list]
        )))
        colors_generator = plt.cm.rainbow(
            np.linspace(0, 1, len(unique_classes)))
        class_to_color = {
            class_name: class_color
            for class_name, class_color in zip(
                unique_classes, colors_generator)
        }
        class_to_color['HC'] = 'olivedrab'
        class_to_color['healthy'] = class_to_color['HC']
        class_to_color['HC_test'] = class_to_color['HC']
        class_to_color['health_ad'] = class_to_color['HC']
        class_to_color['HC_train'] = 'steelblue'
        class_to_color['AD'] = 'indianred'
        class_to_color['AD_test'] = class_to_color['AD']
        class_to_color['AD_train'] = 'goldenrod'
        class_to_color['MCI'] = 'black'
        class_to_color['health_mci'] = class_to_color['MCI']

        pca_fitted = []
        if decomposition is None:
            decomposition = [sklearn.decomposition.PCA(
                n_components=2,
                random_state=0,
            ) for _ in embedding_tensors]
        for embedding_tensor, embeddings_list, pca in zip(
            embedding_tensors,
            all_embeddings_list,
            decomposition,
        ):
            embeddings_for_pca = np.stack(embeddings_list, 0)
            try:
                data_2d = pca.transform(embeddings_for_pca)
            except NotFittedError:
                pca = pca.fit(embeddings_for_pca)
                data_2d = pca.transform(embeddings_for_pca)
            pca_fitted.append(pca)

            print('PCA of %d vectors in dim %d (variance_explained=%s)' % (
                len(embeddings_list),
                embeddings_list[0].shape[0],
                pca.explained_variance_ratio_
            ))

            plt.figure(figsize=(10,7))
            for cidx, class_full_name in enumerate(full_classes_list_unique):
                class_name = class_full_name.split('__')[0]
                marker = cname_extract(class_full_name, 'm', 'o')
                color = cname_extract(class_full_name, 'c', None)
                if color is None:
                    color = class_to_color[class_name]
                else:
                    color = custom_colors_per_class[class_name][int(color)]

                d0 = [
                    data_2d[i, 0]
                    for i, cname in enumerate(classes_list)
                    if cname == class_full_name
                ]
                d1 = [
                    data_2d[i, 1]
                    for i, cname in enumerate(classes_list)
                    if cname == class_full_name
                ]
                plt.scatter(
                    d0, d1,
                    c=[color] * len(d0), label=class_name, marker=marker,
                    **plt_scatter_kwargs
                )
            if plt_legend:
                plt.legend()
            plt.title('Embedding PCA: `%s` (%s)' % (embedding_tensor, legend))
            plt.show()
        return pca_fitted
