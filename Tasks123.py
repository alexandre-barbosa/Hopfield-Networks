# -*- coding: utf-8 -*-
"""
NEVR3004 2022 Project: Hopfield Network

Tasks 1, 2 and 3

Alexandre Barbosa

"""

import numpy as np
import matplotlib.pyplot as plt
import random

N = 50 # network size

S = np.zeros(N) # neuron network
V = np.random.choice((-1., 1.), N) # (random) pattern
W = np.zeros((N, N)) # weights matrix

for i in range(N):
    for j in range(N):
        if i != j: # removes self connections
            W[i][j]  = V[i]*V[j]

""" Task 1. Estabilishing the stability of the pattern """

T = 20
t = 0
m = np.zeros(T) # overlap
h = np.zeros(N) # field

while t < T:
    for i in range(N):
        S[i] = V[i]
        m[t] += (S[i] * V[i])/N
        for j in range(N):
            h[i] += W[i][j] * S[j]
        S[i] = np.sign(h[i])
    t += 1 
    
print("Task 1")

# Plotting
    
plt.plot(np.arange(20), m, color="darkslategray", marker='o', linestyle='dashed')
plt.ylabel("Overlap (m)") 
plt.xlabel("Time (t)")
plt.xticks([1, 5, 10, 15, 20])
plt.savefig('Task1.png')
plt.show()

""" Task 2. Retrieving the pattern """

# We can define a function that generalizes the above procedure

def Weights(S, V):
    
    w = np.zeros((N, N)) # weights matrix
    
    for i in range(N):
        for j in range(N):
            if i != j: # removes self connections
                w[i][j]  = V[i]*V[j]
                
    return w

def Hopfield(S, V, T):

    # build  weights matrix  
    w = Weights(S, V)
    
    # evolve over time
    m = np.zeros(T) # overlap
    h = np.zeros(N) # field
    H = np.zeros(T) # energy
    t = 0 # since starting at t=1 makes Python cry
    
    while t < T:
        for i in range(N):
            m[t] += (S[i] * V[i])/N
            for j in range(N):
                h[i] += w[i][j] * S[j]
                H[t] += w[i][j]*S[j]*S[i]
        # randomly pick neuron to update
        k = np.random.randint(N)
        S[k] = np.sign(h[k])
        H[t]*=-0.5
        t += 1
    
    # Plotting
        
    plt.plot(np.arange(T), m, color="darkslategray", marker='o', markersize=1, linestyle='dashed')
    plt.ylabel("Overlap (m)") 
    plt.xlabel("Time (t)")
    plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.savefig('Task2.png')
    plt.show()
    
    plt.plot(np.arange(T), H, color="darkslategray", marker=',', markersize=3, linestyle='solid')
    plt.ylabel("Energy (H)") 
    plt.xlabel("Time (t)")
    plt.savefig('Task2_Energy.png')
    plt.show()
    
# We now start with only 80% of the patterns in a state matching the pattern we want to retrieve
    
initial_matches = random.sample(range(N), int(0.2*N))

S = np.zeros(N) # neuron network
    
for k in range(N):
    if k in initial_matches:
        S[k] = V[k]
    else:
        S[k] = np.random.choice((-1., 1.))
        
print("Task 2")
Hopfield(S, V, 100)

""" Task 3. Storing and retrieving two patterns """

# We can modify the function to accept two initial patterns

def Hopfield(S, V, U, T):

    # build  weights matrix  
    w = Weights(S, V) + Weights(S, U)
    
    # evolve over time
    m = np.zeros(T) # overlap
    h = np.zeros(N) # field
    t = 0 # since starting at t=1 makes Python cry
    
    while t < T:
        for i in range(N):
            m[t] += (S[i] * V[i])/N
            for j in range(N):
                h[i] += w[i][j] * S[j]
        k = np.random.randint(N)
        S[k] = np.sign(h[k])
        t += 1
    
    # Plotting
        
    plt.plot(np.arange(T), m, marker='o', markersize=1, linestyle='dashed')
    plt.ylabel("Overlap (m)") 
    plt.xlabel("Time (t)")

# It is also useful to define a function that generates the initial network state
    
def Network(Pattern, Noise_Pattern, Noise_Percentage):
    
    initial_matches = random.sample(range(N), int((1.-Noise_Percentage/100.)*N))

    S = np.zeros(N) # neuron network
        
    for k in range(N):
        if k in initial_matches:
            S[k] = Pattern[k]
        else:
            S[k] = Noise_Pattern[k]
            
    return S

# and finally, we generate a second (random) pattern
    
U = np.random.choice((-1., 1.), N)

print("Task 3")
    
Hopfield(Network(V, U, 0),  V, U, 200)
Hopfield(Network(V, U, 20), V, U, 200)
Hopfield(Network(U, V, 0),  V, U, 200)
Hopfield(Network(U, V, 20), V, U, 200)

plt.legend(["V (without noise)", "U (without noise)", "V with 20% noise", "U with 20% noise"])
plt.savefig('Task3.png')

""" Task 4. Storing and retrieving many patterns """

# See "Task4.py" for a more general implementation