import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from AdaBound import AdaBoundOptimizer

class ANFIS:

    def __init__(self, n_inputs, n_rules, learning_rate=1e-2):
        self.n = n_inputs
        self.m = n_rules
        self.inputs = tf.placeholder(tf.float32, shape=(None, n_inputs))  # Input
        self.const_mu=tf.constant([-0.002095970790833235, -0.28361815214157104, -0.2790602147579193, -2.098841905593872, 0.926604688167572], dtype=tf.float32, name='const_mu')
        self.const_sigma=tf.constant([-2.702712297439575, 0.5508643388748169, 2.0941407680511475, -1.0594624280929565, -1.568196177482605], dtype=tf.float32, name='const_sigma')
        self.targets = tf.placeholder(tf.float32, shape=None)  # Desired output
        mu = tf.get_variable("mu",initializer=self.const_mu)  # Means of Gaussian MFS
        self.mu=mu
        sigma = tf.get_variable("sigma",
                                initializer=self.const_sigma)  # Standard deviations of Gaussian MFS
        y = tf.get_variable("y", initializer=[[0.24958953261375427, 0.9933842420578003, -0.3460672199726105, 0.9282270073890686, 0.7177143096923828]])  # Sequent centers

        self.params = tf.trainable_variables()

        self.rul = tf.reduce_prod(
            tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs, (1, n_rules)), mu)) / tf.square(sigma)),
                       (-1, n_rules, n_inputs)), axis=2)  # Rule activations
        # Fuzzy base expansion function:
        self.num = tf.reduce_sum(tf.multiply(self.rul, y), axis=1)
        self.den = tf.clip_by_value(tf.reduce_sum(self.rul, axis=1), 1e-12, 1e12)
        self.out = tf.divide(self.num, self.den)
#         self.loss = tf.losses.huber_loss(self.targets, self.out)  # Loss function computation
        # Other loss functions for regression, uncomment to try them:
        self.loss = tf.losses.mean_squared_error(self.targets, self.out)
#         self.loss = tf.sqrt(tf.losses.mean_squared_error(self.targets, self.out))
        # loss = tf.losses.absolute_difference(target, out)
#         self.loss=tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=self.out)
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)  # Optimization step
        # Other optimizers, uncomment to try them:
#         self.optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss)
#         self.optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)
#         self.optimize = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(self.loss)
#         self.optimize = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99).minimize(self.loss)
#         self.optimize = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=0.1).minimize(self.loss)
#         self.optimize = AdaBoundOptimizer(learning_rate=0.01, final_lr=0.1, beta1=0.9, beta2=0.999, amsbound=False).minimize(self.loss)
        self.init_variables = tf.global_variables_initializer()  # Variable initializer
#         self.test=tf.square(tf.subtract(tf.tile(self.inputs, (1, n_rules)), mu))
    def infer(self, sess, x, targets=None):
        if targets is None:
            return sess.run(self.out, feed_dict={self.inputs: x})
        else:
            return sess.run([self.out, self.loss], feed_dict={self.inputs: x, self.targets: targets})

    def train(self, sess, x, targets):
        yp, l, _ = sess.run([self.out, self.loss, self.optimize], feed_dict={self.inputs: x, self.targets: targets})
        return l, yp

    def plotmfs(self, sess):
        mus = sess.run(self.params[0])
        mus = np.reshape(mus, (self.m, self.n))
        sigmas = sess.run(self.params[1])
        sigmas = np.reshape(sigmas, (self.m, self.n))
        y = sess.run(self.params[2])
        xn = np.linspace(-1.5, 1.5, 1000)
        for r in range(self.m):
            if r % 4 == 0:
                plt.figure(figsize=(11, 6), dpi=80)
            plt.subplot(2, 2, (r % 4) + 1)
            ax = plt.subplot(2, 2, (r % 4) + 1)
            ax.set_title("Rule %d, sequent center: %f" % ((r + 1), y[0, r]))
            for i in range(self.n):
                plt.plot(xn, np.exp(-0.5 * ((xn - mus[r, i]) ** 2) / (sigmas[r, i] ** 2)))
                
    def save_model(self, sess, modelPath):
        saver = tf.train.Saver()
        saver.save(sess, modelPath+"/ANFIS-model")
        
        
    def show(self,sess,x): 
        print(sess.run(self.out, feed_dict={self.inputs:x}).shape) 
        print(sess.run(self.out, feed_dict={self.inputs:x}))