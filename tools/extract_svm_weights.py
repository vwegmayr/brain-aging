import tensorflow as tf
import nibabel as nib
import glob


run = '20180508-144706-debug'
export_dir = '/local/dhaziza/data/%s' % run
meta_file = tf.train.latest_checkpoint(export_dir) + '.meta'
export_variable_name = 'classifier/pd_vs_rest/svm_w:0'
ref_image = glob.glob('/local/ERSM/raw/*/T1_brain_2mm.nii.gz')[0]
ref_image = glob.glob('/local/PPMI/_t1_brain_extracted_registered_1mm_sn/I*.nii.gz')[0]
save_variable_to = '%s_svm_w.nii.gz' % run


ref_image_loaded = nib.load(ref_image)
export_variable_shape = [91, 109, 91]


print('INFO: Reference image %s' % ref_image)
sess = tf.Session()
new_saver = tf.train.import_meta_graph(meta_file)
print('[1/*] Meta graph imported')
new_saver.restore(sess, tf.train.latest_checkpoint(export_dir))
print('[2/*] Graph restored')

variables = tf.trainable_variables()
for v in variables:
    if v.name == export_variable_name:
        print('[3/*] Found variable %s' % (str(v)))
        value = sess.run([tf.reshape(v, ref_image_loaded.get_data().shape)])
        print('[4/*] Value computed')
        new_img = nib.Nifti1Image(
            value[0],
            ref_image_loaded.affine,
            ref_image_loaded.header,
        )
        nib.save(new_img, save_variable_to)
        print('[5/*] %s written' % save_variable_to)
