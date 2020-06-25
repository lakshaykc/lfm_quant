# LFM Quant

#### [by Euclidean Technologies, LLC](http://www.euclidean.com)

On a periodic basis, publicly traded companies are required to report fundamentals: financial data such as revenue, operating income, debt, among others. These data points provide some insight into the financial health of a company.

This repository contains a set of deep learning tools for forecasting company future fundamentals from historical fundamentals and other auxiliary data such as historical prices and macro economic data.

## Installation and Setup

Clone repo, setup environment, and install requirements:

```shell 
$ git clone https://github.com/lakshaykc/lfm-quant.git
$ cd lfm-quant
$ export LFM_QUANT_ROOT=`pwd`
```

You may also want to put LFM_QUANT_ROOT in your shell initialization file such .bash_profile so 
that it does not need to be defined every time you start a shell. 
For example, you could run the following from within the lfm-quant directory:

```shell 
$ echo "export LFM_QUANT_ROOT="`pwd` >> ~/.bash_profile
```

## Building Models
You can train lfm quant on a neural network of a particular type and of a
particular architecture with several other hyperparameters on a particular
dataset by first defining all of these things on a config file, and then
specifying that config file as the point of reference when running
`lfm_quant.py`. Consider, for example, how lfm_quant is run on
`open-dataset.dat`, as specified by `config/system-test.conf`:

```shell
$ python scripts/lfm_quant.py --config=config/system-test.conf --train=True
```

This will load the corresponding data file. Model checkpoints  are saved in a directory defined
by `model_dir` parameter in th config file.
A couple of notes about config files:
> * The user can specify a `.dat` file to use through the `--datafile` and the
>   `data_dir` options (note that the latter is `datasets` by default).
> * `financial_fields` is a range of columns, and should be specified as a
>   string joining the first and last columns of the `.dat` file that the user
>   wants to forecast (for example: saleq_ttm-ltq_mrq).
> * `aux_fields` is similarly also a range of columns that is equivalently
>   specified. Note, however, that these fields are strictly features; they are
>   not part of what the model is trained to predict.

## Generating Forecasts
To generate forecasts for the companies in the validation set, `lfm_quant.py`
must be run with the `--train` option set to False. For example:

```shell
$ python scripts/lfm_quant.py --config=config/system-test.conf --train=False
```
This uses checkpoint directory created under `model_dir` for inference and writes resulting
files in the same dir.

## Dataset
Paper [ref] uses data licensed from Compustat North America. The dataset is proprietary 
and cannot be made public. We provide notes in `dataset_api.md` to create a 
module `dataset`. This module converts data from 
source to tensorflow.datatset object type.

## Contributors and Acknowledgement

This repository was developed and is maintained by 
[Euclidean Technologies, LLC](http://www.euclidean.com/). 
Individual core contributors include 
[Lakshay Chauhan](https://github.com/lakshaykc),
[John Alberg](https://github.com/euclidjda) and 
[Zachary Lipton](https://github.com/zackchase) 

## License 

This is experimental software. It is provided under the [MIT license][mit], 
so you can do with it whatever you wish except hold the authors responsible 
if it does something you don't like.

[mit]: http://www.opensource.org/licenses/mit-license.php



