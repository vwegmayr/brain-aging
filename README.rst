Setup instructions
------------------

1. Setup packages
2. Generate data according to instructions below
3. Run the code with :code:`python run.py --config configs/example_config.yaml -a fit`

Packages setup
--------------

1. Some required pip packages

.. code-block:: shell

  pip install scipy pandas numpy tensorflow-gpu

Data generation
---------------
1. Run :code:`data/generate.sh`
2. Install and setup `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki>`_

Then, depending on the data you need:

- :code:`/local/ADNI_AIBL`: see `MRI-Fusion <https://gitlab.vis.ethz.ch/ise-squad/mri-fusion>`_ project
- :code:`/local/PPMI`: :code:`python run.py --config configs/mri_pipeline/ppmi_t2.yaml -a transform`
- :code:`/local/KOLN`:

  1. Extract KOLN T2 images to :code:`/local/KOLN/T2/raw`. This will create files such as :code:`/local/KOLN/T2/raw/1397P/140225/1397_t2.nii.gz`
  2. :code:`python run.py --config configs/mri_pipeline/koln_t2.yaml -a transform`

3. Finally, generate some CSV files required by running :code:`notebooks/convert_csv.ipynb`

Contribute (Sumatra setup for experiments tracking)
---------------------------------------------------
.. code-block:: shell

  pip install --upgrade git+https://gitlab.vis.ethz.ch/vwegmayr/sumatra.git
  pip install gitpython
  smt init -d data -i data -e python -m run.py -c error -l cmdline feature-robustness
