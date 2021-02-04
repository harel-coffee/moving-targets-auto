import argparse
from source.validation import Validation


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmarks on classical datasets')

    # Configure options

    parser.add_argument('--dataset',
                        choices=['iris', 'redwine', 'whitewine', 'shuttle', 'dota2', 'crime', 'adult'],
                        help='The dataset to be used')

    parser.add_argument('--mtype',
                        choices=['fairness', 'balance'],
                        help='The type of Master problem to be solved (not meaningful for cvx learner)')

    parser.add_argument('--ltype',
                        default='lr',
                        help='The type of Learner (lr = Logistic Regression, ' +
                                'rf = Random Forest, lbrf = Low Bias RF, ' +
                                'nn = Neural Network, sbrnn = NN + SBR, ' +
                                'cvx = LR + SBR, cnd = Kamiran and Calders' +
                                'tfco = Google Method)')

    parser.add_argument('--alpha',
                        dest='alpha_',
                        type=float,
                        default=0,
                        help='The losses weight for the iterative method (default is 0)')

    parser.add_argument('--beta',
                        dest='beta_',
                        type=float,
                        default=0,
                        help='The ball radius for the iterative method (default is 0)')

    parser.add_argument('--iterations',
                        type=int,
                        default=1,
                        help='Number of iterations for the iterative method (default is 0)')

    parser.add_argument('--initial_step',
                        type=str,
                        choices=['pretraining', 'projection'],
                        default='pretraining',
                        help='Define the initial step of the algorithm: pretraining or projection '
                             '(default is pretraining).')

    parser.add_argument('--nfolds',
                        type=int,
                        default=5,
                        help='Number of folds for randomized evaluation (default is 5)')

    parser.add_argument('--use_prob', action='store_true',
                        help='Use the class probabilities instead ')

    # Parse arguments
    args = parser.parse_args()

    instance = Validation(args.dataset, args.nfolds, args.mtype, args.ltype,
                          args.iterations, args.alpha_, args.beta_, args.initial_step, args.use_prob)
    # Validation
    instance.validate()
    instance.collect_results()

    # Test
    instance.test()
    instance.collect_results()
