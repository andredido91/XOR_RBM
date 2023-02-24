#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 00:13:54 2022

@author: andrea
"""

from __future__ import print_function
import numpy as np
import timeit
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import sys

def tf_xavier_init(fan_in, fan_out, const=1.0, dtype=np.float32):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)


def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))   #relu is the activation function 0 if input <0 and 1 if input >= 0


def sample_gaussian(x, sigma):
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)

#j=0
# if j=0
#    print()
# j=j+1
#



class RBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=5,
                 momentum=0,
                 xavier_const=1.0,
                 err_function='mse',
                 use_tqdm=False,
                 # DEPRECATED:
                 tqdm=None):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None

        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.neuron_index = tf.placeholder(tf.int32)
        
        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])

        self.w = tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden, const=xavier_const), dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None
                
        self._initialize_vars()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None

        if err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        else:
            self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))



        init = tf.global_variables_initializer()            #   https://danijar.com/what-is-a-tensorflow-session/
        self.sess = tf.Session()                            #   tf.Session(run(tf.global_variables_initializer()))
        self.sess.run(init)                                 #   this expression initialize a tensorflow session.
                                                            #   The session will also allocate memory to store the current value of the variable.
                                                            #   the value of our variable is only valid within one session.

    def _initialize_vars(self):
        pass

    def get_err(self, batch_x, verbose = False):
        if verbose:
            print("x: ",self.sess.run(self.x, feed_dict={self.x: batch_x}),
                "compute_visible: ",self.sess.run(self.compute_visible, feed_dict={self.x: batch_x}))
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_energy(self, data):
        return self.sess.run(self.compute_energy, feed_dict={self.x: data})

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})   #while exploiting compute_visible, the x variable acquire the value of batch_x

    def partial_fit(self, batch_x):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})

    def get_prob(self, visible_neuron_index,batch_x):
        return self.sess.run(self.getprob_visible, feed_dict={self.neuron_index: visible_neuron_index,self.x: batch_x})

    def reset(self):
        return tf.reset_default_graph()
    
    def fit(self,
            data_x,
            n_epoches=10,
            batch_size=10,
            learning_rate=5,
            decay = 0,
            shuffle=True,
            verbose=True,
            epochs_to_test = 1,
            early_stop = False):
        assert n_epoches > 0

        self.learning_rate = learning_rate

        n_data = data_x.shape[0]        # get information over the number of instances in the data array for example if dataset = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]]) we get dataset.shape[0]=4
                                        # in our case n_data = 1 in learning an 10 in predicting
        #j=0
        #if j==0:
        #    print(n_data)
        #j=j+1

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)       #in our case n_data is composed by 10 instance so n_batches is equal to 1
            #print(batch_size)
            #print(n_data)
            #print(n_batches)
        else:
            n_batches = 1

        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)    # create a tuple of iterable integer from 0 to n_data (so to the number of train instance)
        else:
            data_x_cpy = data_x

        errs = []

        delta_energies = []

        sample = self.reconstruct(np.random.rand(1,self.n_visible))[0]>=0.5     # tf.Session().run(init)

        if hasattr(self, 'image_height'):
            plt.figure()
            plt.axis('off')
            plt.title("Image reconstructed before training ", y=1.03)

            plt.imshow(sample.reshape(self.image_height, -1))


        for e in range(n_epoches):
            epoch_errs = np.zeros((n_batches,))             #a vector of 10 zeros that will host the error of each batch
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]               #allow to shuffle (change the instance order) inside the data vector

            r_batches = range(n_batches)                    # create a tuple which span over the number of batches

            if verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

################################Training of the net ################################

            for b in r_batches:                                                 #we have only one batch of 10 instance in our case
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]       #it would recover the n-th batches if it was composed by more that 10 instances
                self.partial_fit(batch_x)
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if verbose and e % 1000 == 0:
                err_mean = epoch_errs.mean()

                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])            # put errs and epoch_errs together in a vector by "avvicinare verticalmente i due vettori"
            self.learning_rate *= (1-decay)

            sample = self.reconstruct(np.random.rand(1,self.n_visible))[0]>=0.5

            if e%20000 == 0:
                pass


        return errs

    def predict(self, data, positions_to_predict):
        data = np.array(data)
        min_energy = 10000000
        best_answer = []
        need_to_predict = len(positions_to_predict)
        #print("need to predict= ",need_to_predict)
        total_possibilities_num = 2**need_to_predict

        data = np.repeat(data,total_possibilities_num,axis=0)
        #print("data= ", data)
        for possibility_idx, possibles in enumerate(range(total_possibilities_num)):    #enumerate(range(...)) = [(0, 0), (1, 1)]
            for idx,possible in enumerate(bin(possibles)[2:]):
                data[possibility_idx, positions_to_predict[idx]]=int(possible)
        #print("All possibilities: ",data)
        energy = self.get_energy(data)
        # print("energy:",energy)
        best_answer_index = np.argmin(energy)
        # print("best_answer_index:",best_answer_index)
        min_energy = energy[best_answer_index]
        best_answer = data[best_answer_index]

        return best_answer[positions_to_predict], energy

    def get_weights(self):
        return self.sess.run(self.w),\
            self.sess.run(self.visible_bias),\
            self.sess.run(self.hidden_bias)

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        saver.restore(self.sess, filename)
        
        
        
class BBRBM(RBM):
    def __init__(self, *args, **kwargs):
        RBM.__init__(self, *args, **kwargs)

    def _initialize_vars(self):
        hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)                                              # samo of compute_hidden
        visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.visible_bias)    # USE sample_bernoulli(hidden_p) OR SIMPLY hidden_p ???????????????
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        self.hidden_p = hidden_p
        self.visible_recon_p = visible_recon_p

        positive_grad = tf.matmul(tf.transpose(self.x), hidden_p)                   #partial fit pass batch_x as x
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        def f(x_old, x_new):                                                        #A backslash at the end of a line tells Python to extend the current logical line over across to the next physical line.
            return self.momentum * x_old +\
                   self.learning_rate * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])       #in our case momentum is zero

        delta_w_new = f(self.delta_w, positive_grad - negative_grad)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0))
        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias)
        self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias)

        # https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf (25)
        self.input_to_h = tf.matmul(self.x, self.w) + self.hidden_bias # x
        self.compute_energy = - tf.reduce_sum(tf.multiply(self.x, self.visible_bias),axis=1) - tf.reduce_sum(tf.log(1+tf.exp(self.input_to_h)),axis=1)
                                    #multiply perform elementwise multiplication of tensor element and return a tensor
                                    #reduce_sum perform a sum over axis and return a tensor of the same dimension (row number) and only one axis
                                    #https://www.tensorflow.org/guide/tensor
                                    
        def g(l):
            self.summation=0
            for i in range(self.n_hidden):
                self.summation +=self.w[l][i]*self.compute_hidden[0,i]
                print(self.hidden_p)
            return self.summation
                               
        self.getprob_visible=tf.nn.sigmoid(self.visible_bias[self.neuron_index]+g(self.neuron_index))     
        
###############################
#####Beginning of Training#####
###############################
                          
bm = BBRBM(n_visible=3,n_hidden=6)


###################################################
# Train and predict same data in same iterations###
###################################################

# x is the correct data to train the model
# x2 has wrong xor answer ( the third digit )
# x2 would be reconstructed to x after training


# dataset = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])


# for i in range(4):                  # here we train over dataset instance one by one
#     x = dataset[i:i+1].copy()
#     x2 = dataset[i:i+1].copy()
#     x2[0,2] = 1-x2[0,2]             # we build a wrong instance by inverting it's output on the visible neuron [0,2]
#     print ("x:",x)
#     print ("x2:",x2)
#     print ("Training model with x")
#     err = bm.fit(x,n_epoches=1000)
#     print ("model reconstructed x:",np.round(bm.reconstruct(x)))
#     print ("model reconstructed x2:",np.round(bm.reconstruct(x2)))
#     print ("Training done")
#     positions_to_predict = [2]
#     prediction = bm.predict(x,positions_to_predict=positions_to_predict) # best result, energy for all possibilities
#     print ("model predicting x third digit to be:",prediction[0].tolist())
#     prediction = bm.predict(x2,positions_to_predict=positions_to_predict)
#     print ("model predicting x2 third digit to be:",prediction[0].tolist())
#     print ("------------------------------------",end="\n \n")

# bm.reset()

###############################################################
##Majority vote data would be the data reconstruction target###
###############################################################

training_data = np.array([[0,1,1],[0,1,0]])                         # train the net with 9 correct value and one wrong value for 1000 epoch
training_data = np.repeat(training_data, [8,2],axis=0)              # remember that training support shuffle (change the instance order) inside the data vector
print ("training_data:",training_data)
start = timeit.default_timer()
err = bm.fit(training_data,n_epoches=3)
stop = timeit.default_timer()
print ('Time: ', stop - start, "sec")



                ##############################################################################################################################
                ########## Qui verifichiamo la distribuzione di probabilità prevista dalla rete per dei pattern di input######################
                ##############################################################################################################################

#batch_a=bm.transform(np.array([[0,1,1]]))
#print("hidden neuron value for [0,1,0]:  ",batch_a)
#for 
#print("la probabilità di [0,1,1] é: ", batch_a[0,0]*batch_a[0,1]*batch_a[0,2]*batch_a[0,3]*batch_a[0,4]*batch_a[0,5])
#batch_b=bm.transform(np.array([[0,1,0]]))
#print("hidden neuron value for [0,1,0]:  ",batch_b)
#print("la probabilità di [0,1,0] é: ", batch_b[0,0]*batch_b[0,1]*batch_b[0,2]*batch_b[0,3]*batch_b[0,4]*batch_b[0,5])


                ##################################################################################################################################################################
                ###### Qui verifichiamo la probabilità di ogni signolo neurone di input, informazione utile nel caso in cui si voglia usare la rete a scopo di generare dati######
                ##################################################################################################################################################################

for visible_neuron_index in range(3):       #index are 0,1,2 corresponding to input [0,1,#] and output [#,#,2]
    probability= bm.get_prob(visible_neuron_index=visible_neuron_index,batch_x=training_data[0:1])
    print("p(v_"+ str(visible_neuron_index+1)+ "=1|h,W)= : " + str(probability),end="\n")

#visible_neuron_index=3
#probability= bm.get_prob(visible_neuron_index=visible_neuron_index)
#print("p(v_3=1|h,W)= : ",probability,end="\n \n")

print ("model reconstructed [0,1,1]:",np.round(bm.reconstruct(training_data[0:1])))
print ("model reconstructed [0,1,0]:",np.round(bm.reconstruct(training_data[-1:])))
positions_to_predict = [2]
prediction = bm.predict(training_data[0:1],positions_to_predict=positions_to_predict)
print ("model predicting [0,1,1] third digit to be:",prediction[0].tolist())
prediction = bm.predict(training_data[-1:],positions_to_predict=positions_to_predict)
print ("model predicting [0,1,0] third digit to be:",prediction[0].tolist(),end="\n \n \n \n")

#print(bm.get_weights())

bm.reset()
print("Restarting session and train for 3000 epoch instead that only 1000",end="\n \n \n \n")




#######################################################################################################################
####Osserviamo come allenando ulteriormente la rete sia possibile rafforzare l'intera distribuzione di probabilità######
#######################################################################################################################

training_data = np.array([[0,1,1],[0,1,0]])                         # train the net with 9 correct value and one wrong value for 1000 epoch
training_data = np.repeat(training_data, [8,2],axis=0)
print ("training_data:",training_data)
start = timeit.default_timer()
err = bm.fit(training_data,n_epoches=3000)
stop = timeit.default_timer()
print ('Time: ', stop - start, "sec")
print ("model reconstructed [0,1,1]:",np.round(bm.reconstruct(training_data[0:1])))
print ("model reconstructed [0,1,0]:",np.round(bm.reconstruct(training_data[-1:])))

positions_to_predict = [2]
prediction = bm.predict(training_data[0:1],positions_to_predict=positions_to_predict)
print ("model predicting [0,1,1] third digit to be:",prediction[0].tolist())
prediction = bm.predict(training_data[-1:],positions_to_predict=positions_to_predict)
print ("model predicting [0,1,0] third digit to be:",prediction[0].tolist())

for visible_neuron_index in range(3):       #index are 0,1,2 corresponding to input [0,1,#] and output [#,#,2]
    probability= bm.get_prob(visible_neuron_index=visible_neuron_index,batch_x=training_data[0:1])
    print("p(v_"+ str(visible_neuron_index+1)+ "=1|h,W)= : " + str(probability),end="\n")

print("Not Restarting session and start the UNLEARNING PROCESS",end="\n \n \n \n")
#######################################################################################################################################
######### Now we will make the network UNLEARN what it has learned by increasing the number of iteration over a set of WRONG DATA
#######################################################################################################################################

training_data = np.array([[0,1,1],[0,1,0]])                         # train the net with 2 correct value and 9 wrong value for 1000 epoch
training_data = np.repeat(training_data, [2,8],axis=0)
print ("training_data:",training_data)
start = timeit.default_timer()
err = bm.fit(training_data,n_epoches=3000)
stop = timeit.default_timer()
print ('Time: ', stop - start, "sec")
print ("model reconstructed [0,1,1]:",np.round(bm.reconstruct(training_data[0:1])))
print ("model reconstructed [0,1,0]:",np.round(bm.reconstruct(training_data[-1:])))
positions_to_predict = [2]
prediction = bm.predict(training_data[0:1],positions_to_predict=positions_to_predict)
print ("model predicting [0,1,1] third digit to be:",prediction[0].tolist())
prediction = bm.predict(training_data[-1:],positions_to_predict=positions_to_predict)
print ("model predicting [0,1,0] third digit to be:",prediction[0].tolist())

for visible_neuron_index in range(3):       #index are 0,1,2 corresponding to input [0,1,#] and output [#,#,2]
    probability= bm.get_prob(visible_neuron_index=visible_neuron_index,batch_x=training_data[0:1])
    print("p(v_"+ str(visible_neuron_index+1)+ "=1|h,W)= : " + str(probability),end="\n")

bm.reset()


print(" ",end="\n \n \n \n")



print("Restarting session and train all xor configurations",end="\n \n \n \n")
#####################################
###### Try all xor configurations####
#####################################
training_data = np.array([[0,0,0],[1,0,1],[0,1,1],[1,1,0]])

# print "training_data:",training_data
start = timeit.default_timer()

training_data_ = np.repeat(training_data, [1,1,1,1],axis=0)
err = bm.fit(training_data_,n_epoches=2000)

stop = timeit.default_timer()
print ('Time: ', stop - start, "sec")
print ("model reconstructed [0,0,0]:",np.round(bm.reconstruct(training_data[0:1])))
print ("model reconstructed [0,0,1]:",np.round(bm.reconstruct([[0,0,1]])))
print ("model reconstructed [1,0,1]:",np.round(bm.reconstruct(training_data[1:2])))
print ("model reconstructed [1,0,0]:",np.round(bm.reconstruct([[1,0,0]])))
print ("model reconstructed [0,1,1]:",np.round(bm.reconstruct(training_data[2:3])))
print ("model reconstructed [0,1,0]:",np.round(bm.reconstruct([[0,1,0]])))
print ("model reconstructed [1,1,1]:",np.round(bm.reconstruct(training_data[3:4])))
print ("model reconstructed [0,1,0]:",np.round(bm.reconstruct([[1,1,1]])))
                                              
positions_to_predict = [2]
prediction = bm.predict(training_data[0:1],positions_to_predict=positions_to_predict)
print ("model predicting [0,0,0] third digit to be:",prediction[0].tolist())
prediction = bm.predict(training_data[1:2],positions_to_predict=positions_to_predict)
print ("model predicting [1,0,1] third digit to be:",prediction[0].tolist())
prediction = bm.predict(training_data[2:3],positions_to_predict=positions_to_predict)
print ("model predicting ",training_data[2:3]," third digit to be:",prediction[0].tolist())
prediction = bm.predict(training_data[-1:],positions_to_predict=positions_to_predict)
print ("model predicting ",training_data[-1:]," third digit to be:",prediction[0].tolist())




