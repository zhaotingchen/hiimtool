# **hiimtool**
`hiimtool` is a toolkit for dealing with many aspects of 21cm intensity mapping data analysis. It consists of several separate components:

- `hiimtool.grid_util` provides efficient visibility gridding tools for [**casa**](https://casadocs.readthedocs.io/en/stable/) `measurementset` files suitable for parallel computing.
- `hiimtool.hiimvis` deals with 21cm power spectrum estimation in visibility using delay transform techniques.
- `hiimtool.tnsim_util` can be used to generate thermal noise simulations.
- `hiimtool.power_util` includes quadratic estimator for 21cm power spectrum in image space, as well as foreground removal tools.
- `hiimtool.util` contains miscellaneous utility functions for power spectrum estimations.
- `hiimtool.basic_util` contains basic utility functions.
- `hiimtool.config_util` contains function for handling configuration files and generate slurm sbatch scripts for [interim](https://github.com/zhaotingchen/hiim_pipeline/tree/main/interim).
- `hiimtool.ms_tool` deals with measurementset.

This package is still very much under construction, with loads of documentatino and tests missing. See some [examples](examples/) for some basic usage.

## Installation
Clone this repo:
```
git clone https://github.com/zhaotingchen/hiimtool
```

And run
```
cd hiimtool/
pip install -e .
```

Development install is highly recommended, as the package is not in any stable version yet. Note that `hiimtool` uses a variety of packages including [**torch**](https://pytorch.org/) and [**casa**](https://casadocs.readthedocs.io/en/stable/), and it may be hard to configure your environment to use both. The different components of `hiimtool` is deliberately separated from each other and can be used without installation. In that case, instead of running `pip install`, in your python code simply do
```python
import sys
sys.path.append('path/to/hiimtool/src/')
```

## Related Work
Papers that use this package for producing the results:
1. [A first detection of neutral hydrogen intensity mapping on Mpc scales at z≈0.32 and z≈0.44](https://arxiv.org/abs/2301.11943)
2. [Detecting the HI Power Spectrum in the Post-Reionization Universe with SKA-Low](https://arxiv.org/abs/2302.11504)