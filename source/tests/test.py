from source.validation import Validation


def test():
    # Instance.
    dataset = 'crime'
    nfolds = 5
    mtype = 'fairness'
    ltype = 'gb'
    iterations = 2
    alpha = 1
    beta = 1
    initial_step='pretraining'
    use_prob=False

    instance = Validation(dataset=dataset,
                          nfolds=nfolds,
                          mtype=mtype,
                          ltype=ltype,
                          iterations=iterations,
                          alpha=alpha,
                          beta=beta,
                          initial_step=initial_step,
                          use_prob=use_prob)

    instance.validate()
    instance.collect_results()

    instance.test()
    instance.collect_results()


if __name__ == '__main__':
    test()
