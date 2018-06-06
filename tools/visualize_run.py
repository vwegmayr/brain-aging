import tensorflow as tf
import nibabel as nib
import glob
import os
import sys
os.chdir('/local/home/dhaziza/entrack')
sys.path.append('/local/home/dhaziza/entrack/')


from src.deepnn.visualization.network_loader import NetworkLoader
from src.deepnn.visualization.marginal_difference_analysis \
    import MarginalDifferenceAnalysis, PyramidalMDA


def extract_image_id(fname):
    fname = os.path.basename(fname)
    fname = fname.split('.')[0]
    fname = fname.split('_')[0]
    return fname


WORKER_COUNT = 1
WORKER_IDX = 0
run = '20180516-193224'
# run = '20180523-121831'
# run = '20180518-104235'


print('##### WORKER %d / %d [run %s] ######' % (
    WORKER_IDX+1, WORKER_COUNT, run))
network_loader = NetworkLoader('/local/dhaziza/data/%s' % run)
sess = network_loader.sess
mda_dataset = MarginalDifferenceAnalysis.load_dataset(
    network_loader.dataset['test']['health_ad'][0:100] +
    network_loader.dataset['test']['healthy'][0:100],
    sess,
)
probas_tensor = tf.nn.softmax(tf.get_default_graph().get_tensor_by_name(
    'classifier/logits:0'))

pmda = PyramidalMDA(
    session=sess,
    images_dataset=mda_dataset,
    cnn_probas_output=probas_tensor,
    cnn_feed_input='input_features/mri:0',
    cnn_feed_other_values={'is_training:0': False},
    min_depth=2,
    max_depth=2,
    overlap_count=6,
)


all_files_to_visualize = (
    network_loader.dataset['test']['health_ad'][100:] +
    network_loader.dataset['test']['healthy'][100:]
)
# all_files_to_visualize = glob.glob('/local/ADNI_AIBL/processing/_t1_brain_extracted_registered_2mm/I*.nii.gz')

ALL_VISUALIZATIONS_OUT_DIR = '/local/ADNI_AIBL/processing/' + \
    '_visualizations_{run}_mda_d2_ol6/I{image_id}.nii.gz'
for i, image_fname in enumerate(all_files_to_visualize):
    if i % WORKER_COUNT != WORKER_IDX:
        continue
    image_id = extract_image_id(image_fname)
    print('[%d/%d] Visualization for image ID %s' % (
        i+1, len(all_files_to_visualize), image_id))

    if 'aug' in image_fname:
        print('  -- Skipped (augmented image)')
        continue

    image_to_visualize_nifti = nib.load(image_fname)
    image_to_visualize = image_to_visualize_nifti.get_data()

    file_full_name = ALL_VISUALIZATIONS_OUT_DIR.format(
        image_id=image_id,
        run=run,
    )
    if os.path.isfile(file_full_name):
        print('  -- Skipped (already done)')
        continue

    visu = pmda.visualize_image(image_to_visualize)
    new_img = nib.Nifti1Image(
        visu,
        image_to_visualize_nifti.affine,
        image_to_visualize_nifti.header,
    )
    nib.save(new_img, file_full_name)
