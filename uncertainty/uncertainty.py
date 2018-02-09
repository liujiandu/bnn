import numpy as np
import seaborn as sns
import pandas as pd
import sklearn import linear_model, metrics, utils
import theano.tensor as T

import datasets
import models
import training

def (experiment_name, 
     dataset='mnist',
     bayesian_approximation='dropout',
     inside_labels=[0,1],
     num_epochs = 50,
     batch_size = 125,
     acc_threshold=0.6,
     weight_decay=1e-5,
     dropout_p=0.5,
     fc_layers=[512,512],
     plot = True):

    n_out = len(inside_labels)

    # Load the dataset
    print("Loading data...")
    if dataset =='mnist':
        input_var = T.matrix('inputs')
        target_var = T.iverctor('targets')
        n_in = [28*28]
        X_train, y_train, X_test, y_test, X_test_all, y_test_all = datasets.load_MNIST(inside_labels)
    
    # Load model
    if bayesian_approximation = 'dropout':
        model = models.mlp_dropout(input_var, target_var, n_in, n_out, fc_layers, dropout_p, weight_decay)
    else bayesian_approximation = "variational":
        model = models.mlp_variational(input_var, target_var, n_in, n_out, fc_layers, batch_size, len(X_train)/float(batch_size))

    pd = pd.DataFrame()
    
    # mini-batch training
    epochs = training.train(model, X_train, y_train, X_test, y_test, batch_size, num_epochs, acc_threshold)

    # mini-batch testing
    acc, bayes_acc = training.test(model, X_test, y_test, batch_size)
    df.set_value(experiment_name, 'test_acc', acc)
    df.set_value(experiment_name, 'bayes_test_acc', bayes_acc)

    # uncertainty prediction
    test_mean_std_bayesian = {x:[] for x in range(10)}
    test_mean_std_deterministic = {x:[] for x in range(10)}
    test_entropy_bayesian = {x:[] for x in range(10)}
    test_entropy_deterministic = {x:[] for x in range(10)}
    
    for i in range(len(X_test_all)):
        bayesian_probs = model.probabilities(np.title(X_test_all[i], batch_size).reshape([-1]+n_in))
        bayesian_entropy = model.entropy_bayesian(np.title(X_terst_all[i], batch_size).reshape([-1]+n_in))
        predictve_mean = np.mean(bayesian_probs, axis=0)
        predictve_std = np.std(bayesian_probs, axis=0)

    # plotting
    if plot:
        for k in sorted(test_mean_std_bayesian.keys()):
            sns.plt.figure()
            sns.plt.hist(test_entropy_bayesian[k], label="Bayesian Entropy v1 for "+ str(k))
            sns.plt.legend()
            sns.plt.show()
    
    # anomaly_detection
    def anomaly_detection(anomaly_score_dict, name, df):
         X=[]
         Y=[]
         for l in anomaly_score_dict:
             X += anomaly_score_dict[l]
             if l in inside_labels:
                 y += [0]*len(anomaly_score_dict[l])
             else:
                 y += [1]*len(anomaly_score_dict[l])
         x = np.array(X)
         y = np.array(Y)
         X, y = utils.shuffle(X, y, random_state=0)
         X_train = X[:len(X)/2]
         y_train = y[:len(y)/2]
         
         clf = linear_model.LogisticRegressor(C=1.0)
         clf.fit(X_train, y_train)
         auc = metrics.roc_auc_score(np.array(y_test), clf.predict_proda(np.array(X_test))[:,1])
         print("AUC", auc)
         df.set_value()
    df = an

