# Moving Targets

Code for reproducing results in ["Teaching the Old Dog New Tricks: Supervised Learning with Constraints"](https://arxiv.org/abs/2002.10766)

In our paper, we propose a novel method to add constraint support in supervised machine learning models.
Our strategy is based on “teaching” constraint satisfaction to the learner via the direct use of a 
state-of-the-art constraint solver.

The method is general-purpose and can address any learner type and constraints, provided these latter lead to a 
solvable constrained optimization problem (see the paper for details).

## Datasets

We provide the dataset used in the paper for the benchmarking with other algorithms, in the folder 
`resources/{dataset_name}`

## Launch example

The paper's experiments can be reproduced by launching the algorithm from  `run.py`

```
python run.py --dataset whitewine --mtype balance --ltype rf --alpha 1 --beta 0.1 --iterations 15
```

For tensorflow_constrained_optimization problems, instead, we can either launch them within the _Moving Targets_
method (script above) or as follows:

```
python source\tfco_bal.py --dataset whitewine
```

The second launch allows to access the results of _last iteration_, _best iteration_ and _stochastic solution_, while
embedding the method within the Moving Targets algorithm will output the solution corresponding to the last iteration 
value.

## Parameters

* dataset = {'iris', 'redwine', 'whitewine', 'shuttle', 'dota2', 'adult', 'crime'}
* mtype = {'balance', 'fairness'}
* ltype = {'lr', 'rf', 'gb', 'nn', 'cvx', 'sbrnn', 'tfco'}
* alpha = any real positive number
* beta = any real positive number
* iterations = any integer number greater or equal than zero.
* nfolds = any integer greater than 2 (default=5)

## Contacts

[fabrizio.detassis2@unibo.it](mailto:fabrizio.detassis2@unibo.it)