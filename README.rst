=============================
Project specific instructions
=============================
smt init
smt configure --main run.py
smt configure --on-changed=store-diff
pip install scipy pandas 

=============================
ETH Machine Learning Projects
=============================

.. _scikit-learn: http://scikit-learn.org/stable/
.. _sklearn-dev-guide: http://scikit-learn.org/stable/developers/index.html
.. _sumatra: https://pythonhosted.org/Sumatra/
.. _miniconda: https://conda.io/docs/install/quick.html
.. _pipeline: ml_project/pipeline.py
.. _gridsearch: ml_project/model_selection.py
.. _`example config`: .example_config.yaml
.. _VirtualBox: https://www.virtualbox.org/
.. _Ubuntu: https://www.ubuntu.com/download/desktop
.. _data: data/
.. _kaggle-cli: https://github.com/floydwch/kaggle-cli
.. _kaggle: https://inclass.kaggle.com/c/ml-project-1
.. _`Kaggle Project 1`: https://inclass.kaggle.com/c/ml-project-1
.. _`Kaggle Project 2`: https://www.kaggle.com/c/ml-project-2
.. _runner: run.py
.. _regression: ml_project/models/regression.py
.. _`feature selection`: ml_project/models/feature_selection.py
.. _models: ml_project/models
.. _`.environment`: .environment
.. _`request access`: https://docs.gitlab.com/ee/user/project/members/index.html#request-access-to-a-project
.. _`invite link 1`: https://www.kaggle.com/t/4e959a86df6a450ea3dad585f71f67d1
.. _`invite link 2`: https://www.kaggle.com/t/db8d6e93b6ff4efba65481d4d9b53297
.. _`More information`: https://drive.google.com/open?id=1UM_osCot4MomlPQu-G83LHt721nSyIJG5t6rO_yLOZU
.. _`General Project Information`: https://drive.google.com/open?id=1NvAqcPzgnTIflpG6BzeAt3dSUlu1JlYJ_2BsQVab6pI

This repository contains the *Python 3.5.3* framework for the practical projects offered
during the *Machine Learning* course at ETH Zurich. It serves two main purposes:

1. Convenient execution of machine learning models conforming with the scikit-learn_ pattern.
2. Structured & reproducible experiments by integration of sumatra_ and miniconda_.


The project description and result submission are hosted by Kaggle:

- `General Project Information`_
- `Kaggle Project 1`_ (`invite link 1`_)
- `Kaggle Project 2`_ (`invite link 2`_)

.. contents::


Purpose
=======

Many brilliant implementations will be created during the projects, so wouldn't it be great to learn from them?

But have you ever tried to read the code of somebody else? If you just shuddered, you know what we are talking about.

We want to take this pain away (or most of it). This framework aims to enable every student to write their code in the same way.

So when you go to another work, you know what structure to expect, and you can instantly start to navigate through it.

For this purpose, we provide a common file structure and an interface to the scikit-learn_ framework. It offers standardized base classes to derive your solutions from.

But to understand a great result, we need more than the code, that produced it. Which data was used as input? How was it processed? What parameters were used?

For this reason, we have included sumatra_ in the framework. It allows you to track, organize and search your experiments.

Ok, now we understand the code and the experiment setup. So let's run their code!

*ImportError: No module named fancymodule*

Sounds familiar? Don't worry, miniconda_ is a central part of the framework, which provides your code an isolated, functional environment to run.

`More information`_ on how the framework functions.

Getting Started
===============

Get Started (Non-Linux)
-----------------------

The project framework has been tested mainly in Linux (Ubuntu) environments. If you
are using Linux already, you can skip forward to Get Started for Linux.

The framework should also work on OS X, but it has not been tested extensively.
OS X users may choose to skip forward to Get started for Linux and OS X.

If you are using Windows, you need to install VirtualBox_ and create an 64-bit Ubuntu_
virtual machine (VM).

Make sure you allocate sufficient RAM (>= 8GB) and disk space (>= 64GB) for the VM.

If you can not choose 64-bit Ubuntu in VirtualBox, you might have to enable
virtualization in your BIOS.

Once your VM is running, open a terminal and install git:

.. code-block:: shell

    sudo apt-get install git

After that, please continue with Getting Started for Linux.

Get Started (Linux and OS X)
-------------------

First you need to install miniconda_ on your system. If you already have Anaconda
installed you can skip this step.

Having installed miniconda, clone the repository and run the setup script:

.. code-block:: shell

    git clone https://gitlab.vis.ethz.ch/vwegmayr/ml-project.git
    cd ml-project
    python setup.py

Get the data
------------

A simple way to download the data is with the kaggle-cli_ tool.
Make sure the environment is activated:

.. code-block:: shell

    source activate ml_project

If you encounter problems with site-packages try:

.. code-block:: shell

    export PYTHONNOUSERSITE=True; source activate ml_project

Then download the data:

.. code-block:: shell

    cd data/
    kg download -c ml-project-1 -u username -p password

Replace :code:`username` with your Kaggle Username and :code:`password` with your Kaggle password.

Experiments
===========

Running an example experiment
-----------------------------

Make sure the environment is activated:

.. code-block:: shell

    source activate ml_project

Make sure you have downloaded the data to the data_ folder, either by using the
kaggle-cli_ tool or from the kaggle_ homepage.

To run an example experiment, simply type

.. code-block:: shell

    smt run --config .config.yaml -X data/X_train.npy -a fit_transform

    >> =========== Config ===========
    >> {'class': <class 'ml_project.models.transformers.RandomSelection'>,
    >> 'params': {'n_components': 1000, 'random_state': 37}}
    >> ==============================

    >> Record label for this run: '20170810-131658'
    >> Data keys are [20170810-131658/RandomSelection.pkl(9b028327c83a153c0824ca8701f3b78a5106071c [2017-08-10 13:17:04]),
    >> 20170810-131658/X_new.npy(b8c093d7c8e13399b6fe4145f14b4dbc0f241503 [2017-08-10 13:17:04])]

The default experiment will reduce the dimensionality of the training data by
selecting 1000 dimensions at random.

Results can be found in timestamped directories :code:`data/YYYYMMDD-hhmmss`, i.e. for the experiment shown above, you would find the results in
:code:`data/20170810-131658`.

It produced two outputs, first the fitted model *RandomSelection.pkl* and second
the transformed training data *X_new.npy*.

To view the experiment record, type :code:`smtweb`:


.. figure:: https://gmkr.io/s/5995a60a4d561e117a4be2c6/0
   :width: 600
   :target: https://gmkr.io/s/5995a60a4d561e117a4be2c6/0

   Example view of an experiment record.

This command will open a new window in your webbrowser, where you can explore
the information stored about the example experiment.

You can choose from different examples in the `example config`_ file.

More details on experiments
---------------------------

Let us consider the above command in more detail:

.. code-block:: shell

    smt run --config .config.yaml -X data/X_train.npy -a fit_transform

* :code:`smt` invokes sumatra_, which is an experiment tracking tool.

* :code:`run` tells sumatra_ to execute the experiment runner_.

* :code:`--config` points to the paramter file for this experiment.

* :code:`-X` points to the input data

* :code:`-a` tells the runner_ which action to perform.

In addition to :code:`--config` experiments, you can run :code:`--model` experiments.

These two flags cover fit/fit_transform and transform/predict, respectively.

The reason is that for fit/fit_tranform you typically require parameters, whereas
for transform/predict you start from a fitted model.

Continuing the example, we can transform the test data, using
the fitted model from before:

.. code-block:: shell

    smt run --model data/20170810-131658/RandomSelection.pkl -X data/X_test.npy -a transform
    >> Record label for this run: '20170810-134027'
    >> Data keys are [20170810-134027/X_new.npy(b33b0e0b794b64e5d284a602f5440620a21cac1c [2017-08-10 13:40:32])]

Again, sumatra_ created an experiment record, which you can use to track input/output paths.

Writing your own models
-----------------------

Derive your models from sklearn base classes and implement the fit/fit_transform/transform/predict functions. For this project, these functions should cover all you ever need to implement.

For instance, if you want to implement smoothing as a precprocessing step, it clearly matched the fit_transform/transform pattern.

We have provided several placeholder modules in models_, where you can put the code. Two simple examples are already included, KernelEstimator in regression_ and RandomSelection in `feature selection`_.

Please do not create *any* new model files or other files or folders, as we want to preserve the common structure.

To make experimenting easier, we provide an interface to the sklearn classes pipeline_ and gridsearch_. Check out the `example config`_ to find out more about how to use them.

Make sure to read the sklearn-dev-guide_, especially the sections *Coding guidelines*,
*APIs of scikit-learn objects*, and *Rolling your own estimator*.

Furthermore, try to look at the sklearn source code - it is very instructive. You will spot many more of the sklearn utilities!

If you add new packages to your code, please include them in the `.environment`_ file, so that it is available when other people build your environment.

If you think something is missing or should be changed, please contact us via the Piazza forum_ or start an issue on gitlab.

Debugging without Sumatra
-------------------------

If you only want to check if your code runs without invoking sumatra and without
saving outputs, you can simply run

.. code-block:: shell

    python run.py [-h] [-c CONFIG] [-m MODEL] -X X [-y Y] -a {transform,predict,fit,fit_transform}

Use this for debugging only, otherwise your experiments remain untracked and unsaved!

Submission
==========

Code Submission
---------------

It is required to publish your code shortly after the kaggle submission deadline
(kaggle submission deadline + 24 hours).

Make sure you `request access`_ in time, so that you can create a new branch and push code.

First, you have to make sure that your code passes the flake8 tests.
You can check by running

.. code-block:: shell

    flake8

in the ml-project folder. It will return a list of coding quality errors.

Try to run it every now end then, otherwise the list of fixes you have to do before submission may get rather long.

Make sure that your Sumatra records are added:

.. code-block:: shell

    git add .smt/

Next, create and push a new branch which is named :code:`legi-number/ml-project-1`, e.g.

.. code-block:: shell

    git checkout -b 17-123-456/ml-project-1
    git push origin 17-123-456/ml-project-1

The first part has to be your Legi-Number, the number in the second part identifies the project.

This repository runs an automatic quality check, when you push your branch.
Additionally, the timestamp of the push is checked.

Results are only accepted, if the checks are positive and submission is before the deadline.

.. figure:: https://gmkr.io/s/5995a0c7022cf3566f9c65c5/0

    Check under *Pipelines*, if your commit passed the check.
    The *latest* flag indicates which commit is the most current.

Result Submission
-----------------

To submit a prediction (y_YYMMDD-hhmmss.csv), e.g. to get the validation score, you can use
the kaggle-cli_ tool:

.. code-block:: shell

    kg submit data/YYMMDD-hhmmss/y_YYMMDD-hhmmss.csv -c ml-project-1 -u username -p password -m "Brief description"

To view your submissions, just type

.. code-block:: shell

    kg submissions

which will list all your previous submissions. To set a default username, password and project:

.. code-block:: shell

    kg config -u username -p password -c competition

Please note, you have to explicitly select your final submission on Kaggle (`here <https://inclass.kaggle.com/c/ml-project-1/submissions>`_).

Otherwise, Kaggle will automatically select the submission with the best validation score.

Questions & Issues
==================

.. _forum: www.piazza.com/ethz.ch/fall2017/252053500l

Please post general questions about the machine learning projects to the dedicated
Piazza forum_.

For suggestions and problems specifically concerning the project framework, please
open an issue here on gitlab.

If you want to discuss a problem in person, we will offer a weekly project office hour (tbd).
