## Word-level Language Modeling

### Overview

In word-level language modeling tasks, each element of the sequence is a word, where the model
is expected to predict the next incoming word in the text.

## Training

The following configurations were used to obtain the results from the paper, which includes a hyper-parameter optimisation of the gradient descent optimiser:
Each command will train the model on the dataset, save snapshots during training, and evaluate the best found model on the test set.

* TCN trained on PTB dataset

```
python train.py --cuda
```

* Seq-U-Net trained on PTB dataset

```
python train.py --cuda --model sequnet --nhid 390
```

## Training without hyper-parameter optimisation

The above training commands take quite long since they involve an additional hyper-parameter search over the gradient descent optimiser settings.
To train the model only a single time with some given hyper-parameters, one can set ``--mode train``  along with the desired values for learning rate (``--lr``) and clipping (``--clip``).

For more information about the different parameter settings, consult the argument parser in the ``train.py`` script.