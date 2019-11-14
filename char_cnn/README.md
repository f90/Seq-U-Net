## Character-level Language Modeling

### Overview

In character-level language modeling tasks, each sequence is broken into elements by characters. 
Therefore, in a character-level language model, at each time step the model is expected to predict
the next coming character. 

## Training

The following configurations were used to obtain the results from the paper, which includes a hyper-parameter optimisation of the gradient descent optimiser:
Each command will train the model on the dataset, save snapshots during training, and evaluate the best found model on the test set.

* TCN trained on PTB dataset

```
python train.py --cuda --nhid 600 --model tcn
```

* Seq-U-Net trained on PTB dataset

```
python train.py --cuda --nhid 390 --model sequnet
```

## Training without hyper-parameter optimisation

The above training commands take quite long since they involve an additional hyper-parameter search over the gradient descent optimiser settings.
To train the model only a single time with some given hyper-parameters, one can set ``--mode train``  along with the desired values for learning rate (``--lr``) and clipping (``--clip``).

For more information about the different parameter settings, consult the argument parser in the ``train.py`` script.