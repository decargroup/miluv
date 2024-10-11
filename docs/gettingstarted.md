---
title: Getting Started
nav_order: 2
usemathjax: true
---

# Dataset setup
To get started, download the MILUV dataset. By default, the devkit expects the data for each experiment is present in **/miluv/data/EXP_NUMBER**, where EXP_NUMBER is the number of the experiment outlined [here](https://github.com/decargroup/miluv/wiki/Experiment-Appendix#summary-of-experiments).

If you wish to change the default data location, be sure to specify the data directory when creating an instance of the MILUV Dataloader.

# Devkit setup and installation
The devkit requires Python 3.8 or greater. To install the devkit and its dependencies, run

    $ pip3 install .
inside the devkit's root directory (~/path/to/project/MILUV). 

Alternatively, run

    $ pip3 install -e .

inside the devkit's root directory, which installs the package in-place, allowing you make changes to the code without having to reinstall every time. 

For a list of all dependencies, refer to ``requirements.txt`` in the repository's root directory.

To ensure setup and installation was completed without any errors, test the code by running
    
    $ pytest
in the root directory. Note that this set of tests will only verify setup and installation when data is placed in the expected default location specified above.
