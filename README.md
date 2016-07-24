# Semi-supervised learning with squared-loss mutual information regularization

The goal of this project is to learn a multi-class probabilistic classifier in a semi-supervised learning setting. 
We observe labeled data {(x_1, y_1), ... (x_l, y_l)}, and a lot of unlabelded data {x_{l+1}, x_{l+2}, ...}. The goal is to make use of the abundant unlabeled data to learn a better classifier. 
In our framework of squared-loss mutual information (SMI) regularization, we learn a classifier such that the 
following two criteria are respected (with some tradeoff):

1. The predictive loss on the class labels is minimized.
2. The squard-loss version of mutual information between the unlabeled data, and the class labels is maximized.

Advantages of this approach are: 1. analytic solution, 2. out-of-sample classification, 3. probabilistic output.

## This repository 
This repository contains a Matlab implementation of the SMI regularization as described in [our paper](http://jmlr.org/proceedings/papers/v28/niu13.pdf).

    Squared-loss Mutual Information Regularization: A Novel Information-theoretic Approach to Semi-supervised Learning
    Gang Niu, Wittawat Jitkrittum, Bo Dai, Hirotaka Hachiya, Masashi Sugiyama
    ICML, 2013

## How to run?
1. Clone or download this repository. You will get the `smir` folder.
2. In Matlab, execute `startup.m` script by running the command `startup`.
3. Run the demo script `demo_smir_msd_cv.m` to see a demonstration on the two-moon dataset.

## Demo script
a

## License
[MIT license](https://github.com/wittawatj/interpretable-test/blob/master/LICENSE).
