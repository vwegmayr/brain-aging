import re
import os
import glob
import subprocess
import xml.etree.ElementTree as ET
from modules.models.utils import custom_print


def xml_elem_unique(root, path):
    e = root.findall(path)
    assert(len(e) == 1)
    return e[0].text


def filters_match(value, filters):
    match_type = {
        'eq': lambda eq: eq == value,
        'gt': lambda f: float(value) > f,
        'lt': lambda f: float(value) < f,
    }
    return all([
        match_type[type](type_value)
        for type, type_value in filters.items()
    ])


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
        filter_xml=None,
        shard=None,
        params={},
    ):
        self.path = path
        self.params = params
        self.all_files = sorted(glob.glob(files_glob))
        self.files = self.all_files
        self.extract_image_id_regexp = re.compile(extract_image_id_regexp)
        self.regexp_image_id_group = regexp_image_id_group
        self.image_id_enabled = None
        if filter_xml is not None:
            self.filter_xml(**filter_xml)
        if shard is not None:
            self.shard(**shard)

    def filter_xml(self, files, xml_image_id, filters):
        self.image_id_enabled = set()
        discarded_count = 0
        for f in glob.glob(files):
            tree = ET.parse(f)
            root = tree.getroot()
            image_id = int(xml_elem_unique(root, xml_image_id))

            pass_all_filters = True
            for filter in filters:
                value = xml_elem_unique(root, filter['key'])
                if not filters_match(value, filter['value']):
                    pass_all_filters = False
                    break
            if pass_all_filters:
                self.image_id_enabled.add(image_id)
            else:
                discarded_count += 1
        custom_print('[filter_xml] %s images discarded' % (discarded_count))

    def shard(self, worker_index, num_workers):
        custom_print('SHARDS: Worker %s/%s' % (worker_index+1, num_workers))
        assert(worker_index < num_workers)
        self.files = [
            f
            for i, f in enumerate(self.all_files)
            if (i % num_workers) == worker_index
        ]

    def transform(self, X=None):
        folders = ['01_brain_extracted', '02_registered']
        for f in folders:
            self._mkdir(f)
        custom_print('Applying MRI pipeline to %s files' % (len(self.files)))
        for i, mri_raw in enumerate(self.files):
            image_id = self.extract_image_id_regexp.match(
                mri_raw,
            ).group(self.regexp_image_id_group)
            image_id = int(image_id)
            custom_print('Image %s/%s [image_id = %s]' % (
                i, len(self.files), image_id))
            if self.image_id_enabled is not None:
                if image_id not in self.image_id_enabled:
                    custom_print('... Skipped')
                    continue

            paths = [mri_raw] + [os.path.join(
                    self.path,
                    folder,
                    'I{image_id}.nii.gz'.format(image_id=image_id),
                )
                for folder in folders
            ]
            self.brain_extraction(paths[0], paths[1])
            self.template_registration(paths[1], paths[2])

    # ------------------------- Pipeline main steps
    def brain_extraction(self, mri_image, mri_output):
        params = self.params['brain_extraction']
        if 'skip' in params:
            return
        if os.path.exists(mri_output) and not params['overwrite']:
            return

        cmd = 'bet {mri_image} {mri_output} {options}'
        cmd = cmd.format(
            mri_image=mri_image,
            mri_output=mri_output,
            options=params["options"],
        )
        self._exec(cmd)

    def template_registration(self, mri_image, mri_output):
        """
        Registers $mri_image to $mri_template template
        """

        params = self.params['template_registration']
        if 'skip' in params:
            return
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

    # ------------------------- Utils and wrappers
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
