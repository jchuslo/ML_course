'''
hw1.py
Author: Jacob Chuslo

Tufts COMP 135 Intro ML
'''


import numpy as np
import math


''' This function splits data into training and testing data at random'''
def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):

    if not 0 <= frac_test <= 1:
        print("\nPlease choose a value for frac_test between 0 and 1\n")
        raise ValueError()


    if random_state is None:
        rnd = np.random
    elif isinstance(random_state, int) or isinstance(random_state, float):
        rnd = np.random.RandomState(random_state)
    else:
        rnd = random_state

    L = np.size(x_all_LF,0)
    N = math.ceil(frac_test*L)
    M = L - N

    arr_rnd = rnd.permutation(x_all_LF)

    x_test_NF = arr_rnd[0:N].copy()
    x_train_MF = arr_rnd[N:L].copy()

    return x_train_MF, x_test_NF


####################################


'''This data calculates the k nearest neighbors of 2 datasets'''
def calc_k_nearest_neighbors(data_NF, query_QF, K=1):

    if K>np.size(data_NF,0) or (K <= 0):
        print("Please choose an integer K>=1, where K is less than or equal to the dimension of your DATA array")
        raise ValueError

    if np.size(data_NF,1) != np.size(query_QF,1):
        print("The dimensions of the data and query arrays are not equal")
        raise ValueError

    neighb_QKF = np.empty([np.size(query_QF,0), K,np.size(data_NF,1)])
    dict = {}

    for i in range(np.size(query_QF,0)):

        for j in range(np.size(data_NF,0)):
            num = math.sqrt( np.sum( ( np.subtract(query_QF[i], data_NF[j]) )**2 ) )
            dict.update( {(i,j) : num } )

        sorted_dict = sorted(dict, key=dict.get, reverse=False)
        k_sorted_dict = sorted_dict[:K]

        for m in range(0,K):
            for n in range(0,np.size(data_NF,1)):
                neighb_QKF[i,m,n] = data_NF[k_sorted_dict[m][1],n]

        dict.clear()

    return neighb_QKF
