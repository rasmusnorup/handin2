from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os
import argparse
from IPython.display import display
from h2_util import load_train_data, load_test_data
from model_stats import export_dataframe, model_stats


def save_model(cv, name):
    """ Simple save the SVC object learned with cross validation, ignore if you do not need it

    Params:
      cv: GridSearchCV fitted on data
      name: str - name to give model, i.e. rbf_kernel or something like that
    """
    print('saving', name)
    np.savez_compressed(os.path.join('model_weights','{0}_best_model.npz'.format(name)), res=cv.best_estimator_)

def load_model(name):
    """ Simple function to load an SVM  model, ignore if you do not need it

    Args:
      name: str - name given to model when saved
    """
    tmp = np.load(os.path.join('model_weights','{0}_best_model.npz'.format(name)))
    return tmp['res'].item()


### YOUR CODE HERE
### END CODE

if __name__=="__main__":
    """
    Main code you can use and update as you please if you want to use command line arguments
    Otherwise you are free to ignore it.

    There are some extra functions in model_stats you can use as well if you would like to.
    """
    if not os.path.exists('results'):
        print('create results folder')
        os.mkdir('results')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    [au_train_images,au_train_labels] = load_train_data()
    [au_test_images,au_test_labels] = load_test_data()
    rp = np.random.permutation(au_train_labels.size)
    digs = au_train_images[rp,:]
    labs = au_train_labels[rp]
    digs = digs[0:1000, :]
    labs = labs[0:1000]

    parser = argparse.ArgumentParser()
    parser.add_argument('-lin', action='store_true', default=False)
    parser.add_argument('-poly2', action='store_true', default=False)
    parser.add_argument('-poly3', action='store_true', default=False)
    parser.add_argument('-rbf', action='store_true', default=False)
    args = parser.parse_args()
    if args.lin:
        print('running linear svm')
        ### YOUR CODE HERE
        clf = svm.SVC(kernel='linear',C=1)
        clf.fit(digs,labs)
        ### END CODE
    if args.poly2:
        print('running poly 2 svm')
        ### YOUR CODE HERE
        clf = svm.SVC(kernel='poly',degree=2,C=1)
        clf.fit(digs,labs)
        ### END CODE
    if args.poly3:
        print('running poly 3 svm')
        #### YOUR CODE HERE
        clf = svm.SVC(kernel='poly' , degree=3 ,C=1)
        clf.fit(digs,labs)
        ### END CODE
    if args.rbf:
        print('running rbf svm')
        ### YOUR CODE HERE
        clf = svm.SVC(kernel='rbf',C=1)
        clf.fit(digs,labs)

        clf = svm.SVC(kernel='rbf')
        parameters = {
            'C': [1, 10, 100],
            'gamma': [0.01, 0.1, 1]
            }
        gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(X, Y)
        gs_clf = gs_clf.cv_results_
        C = gs_clf['param_C']
        gamma = gs_clf['param_gamma']
        test_scores = gs_clf['mean_test_score']
        train_scores = gs_clf['mean_train_score']
        print("C:\t\t", C)
        print("gamma:\t\t", gamma)
        print("test_acc:\t", test_scores)
        print("train_acc:\t", train_scores)
        ashkdhj
        dataframe = pd.DataFrame(gs_clf)
        relevant = dataframe.filter(['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score', 'param_C', 'param_gamma', 'mean_fit_time']).sort_values(['mean_test_score'])
        display(relevant)

        ### END CODE
