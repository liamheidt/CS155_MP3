########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np
import numpy.matlib
import math
from time import time
import pandas as pd
import datetime

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]      
             
        A_start = self.A_start
        A_start_arr = np.array(A_start)
        # Observation as an array
        O = self.O
        O_arr = np.array(O)
        
        # Transition as an array
        A = self.A
        A_arr = np.array(A)
        
        # Convert probs and seqs to numpy array
        L = self.L
        probs = np.empty((L,M))
        seqs = np.empty((L,M))
        
        probs[:,0] = A_start_arr * O_arr[:, x[0]]
        seqs[:,0] = 0
        
        for i in range(1,M):
            probs[:,i] = np.max(probs[:,i-1] * A_arr.T * O_arr[:, x[i]].reshape(-1,1),1)
            seqs[:,i] = np.argmax(probs[:,i-1] * A_arr.T,1)
        
        xout = np.empty(M, 'B')
        
        xout[-1] = np.argmax(probs[:, M-1])
        
        for i in reversed(range(1,M)):
            xout[i - 1] = seqs[xout[i],i]
            
        x2 = xout.tolist()
        max_seq = ' '.join([str(elem) for elem in x2])

        return max_seq
        


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Transition as an array
        A = self.A
        A_arr = np.array(A)
 
        # Observation as an array
        O = self.O
        O_arr = np.array(O)
        x = np.array(x)
        
        A_start = self.A_start
        A_start_arr = np.array(A_start)
        
        def forward_2(x, A_arr, O_arr, initial_distribution):
            alpha = np.zeros((M, self.L))
            alpha[0, :] = A_start_arr * O_arr[:, x[0]]
 
            for t in range(1, x.shape[0]):
                #for j in range(A_arr.shape[0]):
                #    alpha[t, j] = alpha[t - 1].dot(A_arr[:, j]) * O_arr[j, x[t]]
                    
                alpha[t, :] = alpha[t - 1].dot(A_arr[:, :]) * O_arr[:, x[t]]
     
            return alpha
 
        alpha = forward_2(x, A_arr, O_arr, A_start_arr)
        
        #if normalize == True:
        #    for i in range(0,len(alpha)):
        #        if np.sum(alpha[i,:]) == 0:
        #            alpha[i,:] = alpha[i,:]*0
        #        else:  
        #            alpha[i,:] = alpha[i,:]/np.sum(alpha[i,:])
        
        if normalize == True:
            sum_alphas = np.sum(alpha, axis = 1)
            for i in range(0,len(alpha)):
                if sum_alphas[i] == 0:
                    alpha[i,:] = 0
                else:  
                    alpha[i,:] = alpha[i,:]/sum_alphas[i]
                    
        return alpha


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Transition as an array
        A = self.A
        A_arr = np.array(A)
 
        # Observation as an array
        O = self.O
        O_arr = np.array(O)
        x = np.array(x)
        beta = np.zeros((M+1, A_arr.shape[0]))
        
        def backward_2(x, A_arr, O_arr):
        
            beta = np.zeros((M+1, A_arr.shape[0]))

                    # setting beta(T) = 1
            beta[M] = np.ones((A_arr.shape[0]))
             
            for t in range(M-1, -1, -1):
                #for j in range(A_arr.shape[0]):
                #    beta[t, j] = (beta[t + 1] * O_arr[:, x[t]]).dot(A_arr[j, :])

                beta[t, :] = (beta[t + 1] * O_arr[:, x[t]]).dot(A_arr.T[:, :])
                    
                    
            return beta
 
        beta = backward_2(x, A_arr, O_arr)
        
        #if normalize == True:
        #    for i in range(0,len(beta)):
        #        if np.sum(beta[i,:]) == 0:
        #            beta[i,:] = beta[i,:]*0
        #        else:
        #            beta[i,:] = beta[i,:]/np.sum(beta[i,:])  
           
                    
        if normalize == True:
            sum_betas = np.sum(beta,axis=1)
            for i in range(0,len(beta)):
                if sum_betas[i] == 0:
                    beta[i,:] = 0
                else:
                    beta[i,:] = beta[i,:]/sum_betas[i]
                    
                    
        return beta


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.


        L = self.L
        D = self.D
   
        for a in range(0,L):
            for b in range(0,L):
                acum = 0
                acum_2 = 0
                
                for j in range(0,len(Y)):
                    for i in range(0,len(Y[j])-1):
                        acum = acum + (Y[j][i+1] == b) * (Y[j][i] == a)
                        acum_2 += (Y[j][i] == a)
                        
                        
                self.A[a][b] = acum/acum_2
                
        for z in range(0,L):
            for w in range(0,D):
                acum = 0
                acum_2 = 0
                
                for j in range(0,len(X)):
                    for i in range(0,len(X[j])):
                        acum = acum + (X[j][i] == w) * (Y[j][i] == z)
                        acum_2 += (Y[j][i] == z)
                        
                        
                if acum_2 == 0:
                    self.O[z][w] = 0
                else:
                    self.O[z][w] = acum/acum_2
                
        O1 = np.array(self.O)

        pass


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        L = self.L
        D = self.D      

        for i_iter in range(0,N_iters):
            
            
            A_num = np.zeros((L,L))
            A_den = np.zeros((L,L))
            O_num = np.zeros((L,D))
            O_den = np.zeros((L,D))
        
            print(i_iter)
            
            
            for n_X in range(0,len(X)):
            #for n_X in range(1):
                alfa_m = self.forward(X[n_X],True)
                beta_m = self.backward(X[n_X],True)
    
            # First Marginal Pseudocode
            
                #for i in range(0,len(X[n_X])):
                    #acum_b = 0
                    
                    #for z in range(0,L):                       
                        #for z_p in range(0,L):
                            #acum_b = acum_b + alfa_m[i][z_p]*beta_m[i+1][z_p]
 
                        #P[i][z] = alfa_m[i][z]*beta_m[i+1][z]/acum_b
                
            # First Marginal Vectorized

                P = np.zeros((len(alfa_m),len(alfa_m[0])))  
                
                len_1 = len(X[n_X])
                
                fin_mm = np.multiply(alfa_m,np.delete(beta_m,0,0))
                
                acum_mm = np.sum(fin_mm,axis = 1)
                acum_mm = np.expand_dims(acum_mm,axis = 1)
                
                
                acum_mm[acum_mm == 0] = math.inf
                                
                P[:,:] = fin_mm/acum_mm
                   
                
            # Second Marginal Pseudocode
                #for i in range(1,len(X[n_X])):               
                    #acum_b2 = 0                    
                    #for b in range(0,L):
                        #for a in range(0,L):
                            #for b_p in range(0,L):
                                #for a_p in range(0,L):
                                #    acum_b2 = acum_b2 + alfa_m[i-1][a_p]*self.A[a_p][b_p]*self.O[b_p][X[n_X][i]]*beta_m[i+1][b_p]        
                                

#                           P_2[i][a][b] = alfa_m[i-1][a]*self.A[a][b]*self.O[b][X[n_X][i]]*beta_m[i+1][b]/acum_b2

            # Second Marginal vectorization

                P_2 = np.zeros((len_1,L,L))  
    
                A = np.array(self.A)
                O = np.array(self.O)
        
                alfa_exp = np.expand_dims(np.delete(alfa_m,-1,0), axis=2)
                A_exp = np.expand_dims(A,axis = 0)
                
                mul_1 = alfa_exp*A_exp
                
                mul_2 = O.T[:][X[n_X][:]]
                mul_2 = np.expand_dims(np.delete(mul_2,0,0),axis = 1)
                
                mul_3 = beta_m[:][:]
                mul_3 = np.delete(mul_3,0,0)
                mul_3 = np.delete(mul_3,0,0)
                mul_3 = np.expand_dims(mul_3,axis = 1)
                
                fin_mul = np.multiply(np.multiply(mul_1,mul_2),mul_3)
                
                acum_mul = np.sum(np.sum(fin_mul,axis = 2),axis = 1)
                
                acum_mul[acum_mul == 0] = math.inf
                
                acum_mul = np.expand_dims(acum_mul,axis = 1)
                acum_mul = np.expand_dims(acum_mul,axis = 2)


                P_2[1:len(X[n_X]),:,:] = fin_mul/acum_mul
                

                # Pseudocode
                #for a in range(0,L):
                    #for b in range(0,L):
                        #for i in range(0,len(X[n_X])-1):
                            #A_den[a][b] += P[i][a]
                        #for i in range(0,len(X[n_X])):
                            #A_num[a][b] += P_2[i][a][b]
                            
                #for z in range(0,L):
                    #for w in range(0,D):
                        #for i in range(0,len(X[n_X])):
                            #O_num[z,w] += (X[n_X][i] == w) * P[i,z]
                            #O_den[z,w] += P[i,z]
                        
                        
                # Vectorized method


                temp1 = np.repeat(P[:, :, np.newaxis], L, axis=2)
                temp1[-1,:,:] = 0
                A_den[:,:] += np.sum(temp1, axis = 0)                 
                           
                A_num[:,:] += np.sum(P_2[:,:,:], axis=0)      
                                                    
                A_den[A_den == 0] = math.inf
 
                for w in range(0,D):
                    O_num[:,w] += np.sum(np.multiply(X[n_X][:] == np.matlib.repmat(w,1,len(X[n_X])), P[:,:].T) , axis = 1)         
                        
                        
                O_den[:,:] += numpy.matlib.repmat(np.sum(P[:,:], axis=0),D,1).T
                
                O_den[O_den == 0] = math.inf


            self.A = np.divide(A_num,A_den)
            self.O = np.divide(O_num,O_den)
            
        pass


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        emission = []
        states = []
        
        L = self.L
        D = self.D     
        
        O = np.array(self.O)
        A = np.array(self.A)
           
        # Initial state
        states.append(np.random.randint(0,L))
        emission.append(np.random.choice(D,None, True, O[states[0],:]))
        
        for i in range(1,M):
            
            states.append(np.random.choice(L,None, True, A[states[i-1],:]))
            emission.append(np.random.choice(D,None, True, O[states[i],:]))
            
        
        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)
    
    random.seed(2020)
    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
