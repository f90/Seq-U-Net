# Polyphonic Music Modelling 

## Overview

The goal here is to predict the next note given some history of the notes played.
We evaluate performance based on the negative log-likelihood assigned by the model to the current pitches given the previous ones.


## Training

The following configurations were used to obtain the results from the paper, which includes a hyper-parameter optimisation of the gradient descent optimiser:
Each command will train the model on the dataset, save snapshots during training, and evaluate the best found model on the test set.

* TCN trained on Muse dataset

```
python train.py --cuda --dropout 0.2 --ksize 5 --levels 4 --nhid 215 --data Muse --model tcn
```

* Seq-U-Net trained on Muse dataset

```
python train.py --cuda --dropout 0.2 --ksize 5 --levels 4 --nhid 150 --data Muse --model sequnet
```

* TCN trained on Nott dataset

```
python train.py --cuda --dropout 0.2 --ksize 5 --levels 4 --nhid 215 --data Nott --model tcn
```

* Seq-U-Net trained on Nott dataset

```
python train.py --cuda --dropout 0.2 --ksize 5 --levels 4 --nhid 150 --data Nott --model sequnet
```

* TCN trained on JSB dataset

```
python train.py --cuda --dropout 0.5 --ksize 3 --levels 2 --nhid 220 --data JSB --model tcn --hyper_iter 50
```

* Seq-U-Net trained on JSB dataset

```
python train.py --cuda --dropout 0.5 --ksize 3 --levels 2 --nhid 175 --data JSB --model sequnet --hyper_iter 50
```

## Training without hyper-parameter optimisation

The above training commands take quite long since they involve an additional hyper-parameter search over the gradient descent optimiser settings.
To train the model only a single time with some given hyper-parameters, one can set ``--mode train``  along with the desired values for learning rate (``--lr``) and clipping (``--clip``).

For more information about the different parameter settings, consult the argument parser in the ``train.py`` script.

## Note

The training loss is computed differently here compared to the original TCN repository, as we found it to be unstable and a non-standard approach of squashing values manually to [0,1] range before applying the loss.
Instead, we use the standard negative log-likelihood based objective by directly feeding unnormalised model outputs to the ``BCEWithLogitsLoss`` function.