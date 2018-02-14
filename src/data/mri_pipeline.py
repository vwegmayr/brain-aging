import re
import os
import glob
import subprocess
from modules.models.utils import custom_print


class MriPreprocessingPipeline(object):
    """
    This class handles all the MRI preprocessing.
    Given a $path containing a raw folder, it will:
        1. Extract brains in
            '$path/01_brain_extracted/I{image_id}.nii.gz'
        2. Register brains to a template in
            '$path/02_registered/I{image_id}.nii.gz'
    """

    def __init__(
        self,
        path,
        files_glob,
        extract_image_id_regexp,
        regexp_image_id_group,
        shard=None,
        params={},
    ):
        self.path = path
        self.params = params
        self.all_files = sorted(glob.glob(files_glob))
        self.files = self.all_files
        self.extract_image_id_regexp = re.compile(extract_image_id_regexp)
        self.regexp_image_id_group = regexp_image_id_group
        if shard is not None:
            self.shard(**shard)

    def shard(self, worker_index, num_workers):
        custom_print('SHARDS: Worker %s/%s' % (worker_index+1, num_workers))
        assert(worker_index < num_workers)
        self.files = [
            f
            for i, f in enumerate(self.all_files)
            if (i % num_workers) == worker_index
        ]

    def transform(self, X=None):
        self._mkdir('01_brain_extracted')
        self._mkdir('02_registered')
        custom_print('Applying MRI pipeline to %s files' % (len(self.files)))
        for i, mri_raw in enumerate(self.files):
            image_id = self.extract_image_id_regexp.match(mri_raw).group(self.regexp_image_id_group)
            custom_print('Image %s/%s [image_id = %s]' % (
                i, len(self.files), image_id))

            mri_brain_extracted = os.path.join(
                self.path,
                '01_brain_extracted',
                'I{image_id}.nii.gz'.format(image_id=image_id),
            )
            mri_registered = os.path.join(
                self.path,
                '02_registered',
                'I{image_id}.nii.gz'.format(image_id=image_id),
            )
            self.brain_extraction(mri_raw, mri_brain_extracted)
            self.template_registration(mri_brain_extracted, mri_registered)

    def brain_extraction(self, mri_image, mri_output):
        params = self.params['brain_extraction']
        if os.path.exists(mri_output) and not params['overwrite']:
            return

        bias_option = '-B ' if params['bias'] else ''

        cmd = 'bet {mri_image} {mri_output} {bias_option} -f {f} -m -v'
        cmd = cmd.format(
            mri_image=mri_image,
            mri_output=mri_output,
            bias_option=bias_option,
            f=params["frac_intens_thres"],
        )
        self._exec(cmd)

    def template_registration(self, mri_image, mri_output):
        """
        Registers $mri_image to $mri_template template
        """
        params = self.params['template_registration']

        if os.path.exists(mri_output) and not params['overwrite']:
            return

        cmd = 'flirt -in {mri_image} -ref {ref} -out {out} ' + \
            '-cost {cost} -searchcost {searchcost} -v'
        cmd = cmd.format(
            mri_image=mri_image,
            ref=params['mri_template'],
            out=mri_output,
            cost=params['cost'],
            searchcost=params['searchcost'],
        )
        self._exec(cmd)

    def _exec(self, cmd):
        custom_print('[Exec] ' + cmd)
        os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
        os.environ['FSLDIR'] = '/local/fsl'
        subprocess.call(cmd, shell=True)

    def _mkdir(self, directory):
        try:
            os.mkdir(os.path.join(self.path, directory))
        except OSError:
            pass
