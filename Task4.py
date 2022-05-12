# -*- coding: utf-8 -*-
"""
NEVR3004 2022 Project: Hopfield Network

Task 4

Alexandre Barbosa

"""

import numpy as np
import matplotlib.pyplot as plt
import random

N = 100 # network size
M = 3 # number of patterns
T = 500 # number of time steps
load_parameter = M/N # capacity

def Weights(S, V):
    
    N = S.size
    
    w = np.zeros((N, N)) # weights matrix
    
    for i in range(N):
        for j in range(N):
            if i != j: # removes self connections
                w[i][j]  = V[i]*V[j]
                
    return w

def Hopfield(S, N, M, T):

    w = np.zeros((N, N))
    P = np.zeros((M, N))
    for p in range(M):
        P[p] = np.random.choice((-1., 1.), N)   # generate (random) pattern
        w += Weights(S, P[p])                   # build  weights matrix  
    
    # evolve over time
    m = np.zeros((M, T)) # overlap
    d = np.zeros((M, T)) # hamming distance
    h = np.zeros(N) # field
    H = np.zeros(T) # energy
    t = 0 # since starting at t=1 makes Python cry
    
    for p in range(M):
        print(Hamming_Distance(S, P[p]))
    
    while t < T:
        for i in range(N):
            for p in range(M):
                m[p][t] += (S[i] * P[p][i])/N
            for j in range(N):
                h[i] += w[i][j] * S[j]
                H[t] += w[i][j]*S[j]*S[i]
        for p in range(M):
            d[p][t] = Hamming_Distance(S, P[p])
        # randomly pick neuron to update
        k = np.random.randint(N)
        S[k] = np.sign(h[k])
        H[t] *= -0.5
        t += 1
        
    print("\n")
    for p in range(M):
        print(Hamming_Distance(S, P[p]))
    
    # Plotting
        
    for p in range(M):
        #plt.plot(np.arange(T), m[p], marker='o', markersize=2,  linestyle='dashed')
        plt.plot(np.arange(T), m[p],  marker=',', markersize=3, linestyle='solid')
        plt.ylabel("Overlap (m)") 
        plt.xlabel("Time (t)")
    plt.show()
        #plt.xticks([1, 5, 10, 15, 20])
        #plt.show()
        
    for p in range(M):
        #plt.plot(np.arange(T), m[p], marker='o', markersize=2,  linestyle='dashed')
        plt.plot(np.arange(T), d[p],  marker=',', markersize=3, linestyle='solid')
        plt.ylabel("Hamming Distance (m)") 
        plt.xlabel("Time (t)")
    plt.show()
        
    plt.show()
    plt.plot(np.arange(T), H, marker=',', markersize=3, linestyle='solid')
    plt.ylabel("Energy (H)") 
    plt.xlabel("Time (t)")
    #plt.xticks([1, 5, 10, 15, 20])
    

def Energy(S, w):
    H = 0
    N = S.size
    for i in range(N):
        for j in range(N):
            H += w[i][j]*S[i]*S[j]
    H *= -0.5
            
    return H

def Hamming_Distance(Test_Pattern, Stored_Pattern):
    if (len(Test_Pattern) != len(Stored_Pattern)):
        return -1 # error!
    else:
        hamming_distance = 0
        ksi = np.zeros(len(Test_Pattern))
        zeta = np.zeros(len(Test_Pattern))
        for i in range(len(Test_Pattern)):
            # convert (-1, 1) to (0, 1)
            ksi[i] = (Test_Pattern[i] + 1)/2
            zeta[i] = (Stored_Pattern[i] +1)/2
            hamming_distance += ksi[i] * (1 - zeta[i]) + (1 - ksi[i]) * zeta[i]

    return int(hamming_distance)

def Network(Pattern, Noise_Pattern, Noise_Percentage):
    
    N = Pattern.size
    
    initial_matches = random.sample(range(N), int((1.-Noise_Percentage/100.)*N))

    S = np.zeros(N) # neuron network
        
    for k in range(N):
        if k in initial_matches:
            S[k] = Pattern[k]
        else:
            S[k] = Noise_Pattern[k]
            
    return S

S = Network(np.random.choice((-1., 1.), N), np.random.choice((-1., 1.), N), 50)

Hopfield(S, N, M, T)

# plot energy as a function of time
    # verify capacity