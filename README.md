# mlp_stock
Stock price prediction using ensemble MLP in PyTorch.

Train data is the daily change of CISSM (Compositional Index of Shenzhen Stock Market) from 2005/01 to 2015/06, and the test data is from 2015/07 to 2017/05.

## Requirements
* Pytorch
* Numpy
* Pandas
* Matplotlib

## Usage
1. Run `train_net.py` to train a group of MLPs with `sz_train.csv`, saved in `/MLPs`.
2. Run `test_net.py` to predict stock market trend (in `sz_test.csv`) using ensemble MLP.

## Result
The train error rate (black) and test error rate (red) of a single MLP, changing with epoches.
![](https://github.com/melissa135/mlp_stock/blob/master/error_rate.png)

The return sequence (red) if we buy and sell ETF of CISSM according to our ensemble MLP, comparing with CISSM (black).
![](https://github.com/melissa135/mlp_stock/blob/master/asset.png) 


## Tips
