import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

import numpy as np

#Christopher Tegho
#Source https://gist.github.com/rocknrollnerd/c5af642cf217971d93f499e8f70fcb72#file-bayes_by_backprop-py-L180

def nonlinearity(x):
    return tf.nn.relu(x)

def log_gaussian(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - tf.log(tf.abs(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)

def log_gaussian_logsigma(x, mu, logsigma):
    return -0.5 * np.log(2 * np.pi) - logsigma / 2. - (x - mu) ** 2 / (2. * tf.exp(logsigma))

def get_random(shape, avg, std):
    return tf.random_normal(shape, mean=avg, stddev=std)

def func(x):
    return x**2*10.0

if __name__ == '__main__':
    # data
    X = np.arange(-1.,1., 0.01).reshape((-1,1))
    Y = func(X)
    N =90 
    idx = np.random.choice(X.shape[0], N)
    X = X[idx]
    target = np.int32(Y[idx]).reshape(N, 1)

    train_idx, test_idx = train_test_split(np.array(range(N)), test_size=0.05)
    train_data, test_data = X[train_idx], X[test_idx]
    train_target, test_target = Y[train_idx], Y[test_idx]   
    train_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(train_target))
    
    
    # inputs
    x = tf.placeholder(tf.float32, shape = None, name = 'x')
    y = tf.placeholder(tf.float32, shape = None, name = 'y')
    n_input = train_data.shape[1]
    M = train_data.shape[0]
    sigma_prior = tf.exp(-3.0)
    n_samples = 3
    learning_rate = 0.001
    n_epochs = 100

    stddev_var = 0.1
    


    # weights
    # L1
    n_hidden_1 = 200
    W1_mu = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=stddev_var))
    W1_logsigma = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], mean=1.0, stddev=stddev_var)) 
    b1_mu = tf.Variable(tf.zeros([n_hidden_1])) #CHRIS can change
    b1_logsigma = tf.Variable(tf.zeros([n_hidden_1]))

    # L2
    n_hidden_2 = 200
    W2_mu = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=stddev_var))
    W2_logsigma = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], mean=1.0, stddev=stddev_var))
    b2_mu = tf.Variable(tf.zeros([n_hidden_2])) 
    b2_logsigma = tf.Variable(tf.zeros([n_hidden_2])) 

    # L3
    n_output = 1
    W3_mu = tf.Variable(tf.truncated_normal([n_hidden_2, n_output], stddev=stddev_var))
    W3_logsigma = tf.Variable(tf.truncated_normal([n_hidden_2, n_output], mean=1.0, stddev=stddev_var))
    b3_mu = tf.Variable(tf.zeros([n_output])) 
    b3_logsigma = tf.Variable(tf.zeros([n_output])) 



    #Building the objective
    log_pw, log_qw, log_likelihood = 0., 0., 0.

    for _ in xrange(n_samples):

        epsilon_w1 = get_random((n_input, n_hidden_1), avg=0., std=sigma_prior)
        epsilon_b1 = get_random((n_hidden_1,), avg=0., std=sigma_prior)

        W1 = W1_mu + tf.multiply(tf.log(1. + tf.exp(W1_logsigma)), epsilon_w1)
        b1 = b1_mu + tf.multiply(tf.log(1. + tf.exp(b1_logsigma)), epsilon_b1)

        epsilon_w2 = get_random((n_hidden_1, n_hidden_2), avg=0., std=sigma_prior)
        epsilon_b2 = get_random((n_hidden_2,), avg=0., std=sigma_prior)

        W2 = W2_mu + tf.multiply(tf.log(1. + tf.exp(W2_logsigma)), epsilon_w2)
        b2 = b2_mu + tf.multiply(tf.log(1. + tf.exp(b2_logsigma)), epsilon_b2)

        epsilon_w3 = get_random((n_hidden_2, n_output), avg=0., std=sigma_prior)
        epsilon_b3 = get_random((n_output,), avg=0., std=sigma_prior)

        W3 = W3_mu + tf.multiply(tf.log(1. + tf.exp(W3_logsigma)), epsilon_w3)
        b3 = b3_mu + tf.multiply(tf.log(1. + tf.exp(b3_logsigma)), epsilon_b3)

        a1 = nonlinearity(tf.matmul(x, W1) + b1)
        a2 = nonlinearity(tf.matmul(a1, W2) + b2)
        h =  tf.matmul(a2, W3) + b3

        sample_log_pw, sample_log_qw, sample_log_likelihood = 0., 0., 0.


        for W, b, W_mu, W_logsigma, b_mu, b_logsigma in [(W1, b1, W1_mu, W1_logsigma, b1_mu, b1_logsigma),
                                                         (W2, b2, W2_mu, W2_logsigma, b2_mu, b2_logsigma),
                                                         (W3, b3, W3_mu, W3_logsigma, b3_mu, b3_logsigma)]:

            # first weight prior
            sample_log_pw += tf.reduce_sum(log_gaussian(W, 0., sigma_prior))
            sample_log_pw += tf.reduce_sum(log_gaussian(b, 0., sigma_prior))

            # then approximation
            sample_log_qw += tf.reduce_sum(log_gaussian_logsigma(W, W_mu, W_logsigma * 2))
            sample_log_qw += tf.reduce_sum(log_gaussian_logsigma(b, b_mu, b_logsigma * 2))

        # then the likelihood
        sample_log_likelihood = tf.reduce_sum(log_gaussian(y, h, sigma_prior))

        log_pw += sample_log_pw
        log_qw += sample_log_qw
        log_likelihood += sample_log_likelihood

    log_qw /= n_samples
    log_pw /= n_samples
    log_likelihood /= n_samples
    

    batch_size = 10
    n_batches = M / float(batch_size)
    n_train_batches = int(train_data.shape[0] / float(batch_size))
    minibatch = tf.placeholder(tf.float32, shape = None, name = 'minibatch')
    #pi = (2**(n_epochs-minibatch-1))/(2**n_epochs - 1 )
    pi = (2**(n_batches-minibatch-1))/(2**n_batches - 1 )
    #pi = (1. / n_batches)
    objective = tf.reduce_sum(pi * (log_qw - log_pw) - log_likelihood) / float(batch_size)
    



    # updates
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimize = optimizer.minimize(objective)

    a1_mu = nonlinearity(tf.matmul(x, W1_mu) + b1_mu)
    a2_mu = nonlinearity(tf.matmul(a1_mu, W2_mu) + b2_mu)
    h_mu = tf.matmul(a2_mu, W3_mu) + b3_mu
    
    # Test trained model
    #correct_prediction = tf.equal(tf.argmax(h_mu, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))/ float(test_data.shape[0])
    
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for n in range(1000):
        errs = []
	weightVar = []
        for i in xrange(n_train_batches):
            ob = sess.run([objective, optimize, pi, W2_logsigma], feed_dict={            
	    		x: train_data[i * batch_size: (i + 1) * batch_size],
            		y: train_target[i * batch_size: (i + 1) * batch_size],
			minibatch: n})
            errs.append(ob[0])
	    weightVar.append(np.mean(ob[3]))
	    #print ob[2]
    	pred = sess.run(h_mu, feed_dict={x: test_data})
        print("h_mu:{}, pred:{}".format(test_target, pred))    



