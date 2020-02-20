# Accuracy experiments for GIGAQBX FMM translation operators

This repository contains the code and results for the FMM translation operator
experiments described in Appendix B of the paper
*A Fast Algorithm for Quadrature by Expansion in Three Dimensions* (published [here](https://doi.org/10.1016/j.jcp.2019.03.024) and also [available on arXiv](https://arxiv.org/abs/1805.06106)).

## Installation and Usage

The following software is required:

 * a Fortran compiler
 * SciPy
 * Pandas
 * [sumpy](https://github.com/inducer/sumpy)

To build the code, run `./build.sh`, which compiles an auxiliary Python module
for evaluating associated Legendre functions. You can verify that the build
succeeded by running the tests in `./test_l3d.py`.

The script `./translation_accuracy.py` contains the accuracy experiments
described in the paper. Running this script generates a LaTeX report file. By
default, the script uses the saved data in the \*.csv files and does not re-run
the experiments. To re-run the experiments, move/delete the saved \*.csv data
first.
