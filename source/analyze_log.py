import pandas as pd
import numpy as np
import argparse

params = {
    'dataset': 'Dataset',
    'ltype': 'Ltype',
    'iterations': 'Iterations',
    'alpha': 'Alpha',
    'beta': 'Beta',
    'skip pretraining': 'Skip pretraining',
}
#
measures = {
    # Run parameters
    'Iteration': 'Iteration',
    'Pretraining time': 'Pretraining time',
    'Training time': 'Training time',
    'Label adjustment time': 'Label adjustment time',

    # Regression scores.
    'R2 on the train set, for the pretrained model': 'R2 tr pretrained',
    'MSE on the train set, for the pretrained model': 'MSE tr pretrained',
    'R2 on the test set, for the pretrained model': 'R2 ts pretrained',
    'MSE on the test set, for the pretrained model': 'MSE ts pretrained',
    'R2 on the train set, for the adjusted targets': 'R2 tr OPT',
    'MSE on the train set, for the adjusted targets': 'MSE tr OPT',
    'R2 on the train set, for the model': 'R2 tr model',
    'MSE on the train set, for the model': 'MSE tr model',
    'R2 on the test set, for the model': 'R2 ts model',
    'MSE on the test set, for the model': 'MSE ts model',

    # Classification Scores
    'Accuracy on the train set, for the pretrained model': 'Acc tr pretrained',
    'Accuracy on the test set, for the pretrained model': 'Acc ts pretrained',
    'Accuracy on the train set, for the adjusted targets': 'Acc tr OPT',
    'Accuracy on the train set, for the model': 'Acc tr model',
    'Accuracy on the test set, for the model': 'Acc ts model',

    # Fairness Scores.
    'DIDI perc. index in the training set': 'DIDI train',
    'DIDI perc. index in the test set': 'DIDI test',
    'DIDI perc. index in the training set, for the pretrained model': 'DIDI tr pretrained',
    'DIDI perc. index in the test set, for the pretrained model': 'DIDI ts pretrained',
    'DIDI perc. index in the training set, for the adjusted targets': 'DIDI tr OPT',
    'DIDI perc. index in the training set, for the model': 'DIDI tr model',
    'DIDI perc. index in the test set, for the model': 'DIDI ts model',

    # Balanced Scores.
    'Std. Dev of class frequencies in the training set': 'Stddev tr set',
    'Std. Dev of class frequencies in the test set': 'Stddev ts set',
    'Std. Dev of class frequencies in the training set, for the pretrained model': 'Stddev tr pretrained',
    'Std. Dev of class frequencies in the test set, for the pretrained model': 'Stddev ts pretrained',
    'Std. Dev of class frequencies in the training set, for the adjusted targets': 'Stddev tr OPT',
    'Std. Dev of class frequencies in the training set, for the model': 'Stddev tr model',
    'Std. Dev of class frequencies in the test set, for the model': 'Stddev ts model',
}


def analyze_results():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='log')
    parser.add_argument('--refine', action='store_false', default='True')

    args = parser.parse_args()
    path = './'
    filename = args.name

    with open(path+filename+'.log', mode='r') as f:
        lines = f.readlines()

    rows = dict()
    for key in params.keys():
        rows[key] = 0
    for key in measures.keys():
        rows[key] = 0

    columns = list(params.values()) + list(measures.values())
    results = pd.DataFrame(np.full(fill_value=np.nan, shape=(1000, len(columns))), columns=columns)

    # print(lines)
    for line in lines:
        # dt, l = line.split(' - ')
        x = line.strip()
        if ': ' in x:
            name, val = x.split(': ')
            for k in params.keys():
                if k == name:
                    _name = params[name]
                    _row = rows[name]
                    results[_name].iloc[_row] = val
                    print(f'{_name}: {val}')
                    rows[name] += 1
            for k in measures.keys():
                if k == name:
                    _name = measures[name]
                    _row = rows[name]
                    results[_name].iloc[_row] = float(val)
                    print(f'{_name}: {val}')
                    rows[name] += 1

    # Drop rows that contain only Nan values.
    results = results.dropna(axis=0, how='all')
    out_name = path + filename
    out_kpi_name = out_name + '_kpi'

    # Rearrange values in each column to have equal spacing between them.
    if args.refine:
        for c in results:
            nval = len(results[c].dropna())
            nrows = len(results)
            # Check if it is to be redistributed.
            if (nval > 1) & (nval < nrows):
                spacing = int(nrows / nval)
                assert nrows % nval == 0
                # Exclude the first value that is already in the correct position.
                for n in range(nval-1, 0, -1):
                    val = results[c].iloc[n]
                    results[c].iloc[n * spacing] = val
                    results[c].iloc[n] = np.nan

        kpis = results.copy()
        kpis = kpis.dropna(axis=1, how='all')
        kpis = kpis.fillna(method='ffill')

        group_labels = [
            'Dataset',
            'Ltype',
            'Iterations',
            'Alpha',
            'Beta',
            'Skip pretraining',
            'Iteration'
        ]
        print()
        try:
            group_kpi = kpis.groupby(group_labels, as_index=True)
            kpi_df = [group_kpi.mean().reset_index(), group_kpi.std().reset_index()]
            kpi_df = pd.concat(kpi_df, axis=0, ignore_index=True)
            kpi_df.to_csv(out_kpi_name, header=True, index=False, sep=';', decimal='.')
        except Exception as err:
            print(err)
            pass

    # for key, val in results.items():
    #     print("Key " + str(key) + " has " + str(len(val)) + " values.")

    out_df = pd.DataFrame(results)
    # out_df.to_excel(path+filename+'.xlsx', header=True, index=False)
    # print(out_df)
    print(f'Saving results to: {out_name}')
    out_df.to_csv(out_name, header=True, index=False, sep=';', decimal='.')


if __name__ == '__main__':
    analyze_results()
