import numpy as np
from sympy.combinatorics import Permutation
from itertools import combinations

# n: size of the Rubik's cube (n x n x n)
# N: order of the symmetric group
# enuerate(): enumerates the moves that generate the Rubik's cube
def enumerate(n, N):
    turns = set()

    if n % 2 == 0:
        a = n**2 # number of stickers on each face
        b = a//4 # number of 4-clusters on each face
        c = n//2 # side length of 4-cluster
        d = n//2 # side width of 4-cluster
    else:
        a = n**2-1 # number of (labeled, i.e. without middle center) stickers on each face
        b = a//4 # number of 4-clusters on each face
        c = (n+1)//2 # side length of 4-cluster
        d = (n-1)//2 # side width of 4-cluster

    right = [a+b, 2*a+b, 4*a, 5*a, a+2*b, 2*a+2*b, 4*a+b, 5*a+b, 0]
    front = [3*b, 5*a+3*b, 3*a+2*b, 2*a+2*b, 0, 5*a, 3*a+3*b, 2*a+3*b, a]
    up = [0, a, 3*a+b, 4*a+b, b, a+b, 3*a+2*b, 4*a+2*b, 2*a]
    left = [a+3*b, 5*a+2*b, 4*a+2*b, 2*a+3*b, a, 5*a+3*b, 4*a+3*b, 2*a, 3*a]
    back = [b, 2*a, 3*a, 5*a+b, 2*b, 2*a+b, 3*a+b, 5*a+2*b, 4*a]
    down = [2*b, 4*a+3*b, 3*a+3*b, a+2*b, 3*b, 4*a, 3*a, a+3*b, 5*a]

    faces = [right, front, up, left, back, down]

    for face in faces:
        for k in range(0, d):
            P = Permutation(N-1)
            for i in range(0, c):
                P*= Permutation(face[0]+k*c+i, face[1]+k*c+i, face[2]+k*c+i, face[3]+k*c+i)
            for j in range(0, d):
                P*= Permutation(face[4]+j*c+k, face[5]+j*c+k, face[6]+j*c+k, face[7]+j*c+k)
            if k == 0:
                for l in range(0, b):
                    P*= Permutation(face[8]+l, face[8]+l+b, face[8]+l+2*b, face[8]+l+3*b)
            turns.add(P)

    return turns

# n: size of the Rubik's cube (n x n x n)
# human_order(): mimics a human solving technique by choosing a specific order of stabilization
# still needs to be automated
def human_order(n):
    if n == 3:
        return [1,3,5,7,0,2,4,6,9,13,35,39,25,27,29,31,24,26,28,30,
               8,10,11,12,14,15,16,17,18,19,20,21,22,23,32,33,34,36,37,38,40,41,42,43,44,45,46,47]
    if n == 4:
        return [3,7,11,15,51,55,59,63,19,23,27,31,35,39,43,47,67,71,75,79,83,87,91,95,
                1,6,5,10,9,14,13,2,0,4,8,12,17,22,25,30,66,77,69,74,49,54,53,58,57,62,61,50,48,52,56,60,
                20,21,26,24,36,37,42,40,64,65,70,68,80,81,86,84,16,18,29,28,32,34,45,44,72,73,78,76,88,89,94,92,33,38,41,46,82,93,85,90]

# n: size of the Rubik's cube (n x n x n)
# N: order of the symmetric group
# solved_states(): computes the set of visually solved states for a Rubik's cube of size n
def solved_states(n, N):
    solved_states = set()
    solved_states.add(Permutation(N-1))
    helper_set = set()

    if n % 2 == 0:
        a = n**2 # number of stickers on each face
        b = a//4 # number of 4-clusters on each face
        c = n//2 # side length of 4-cluster
        d = n//2 # side width of 4-cluster
    else:
        a = n**2-1 # number of (labeled, i.e. without middle center) stickers on each face
        b = a//4 # number of 4-clusters on each face
        c = (n+1)//2 # side length of 4-cluster
        d = (n-1)//2 # side width of 4-cluster

    for k in range(1, d):
        for i in range(1, c):
            P = Permutation(N-1)(k*c+i, k*c+i+b)
            helper_set.add(P)

    for r in range(1, len(helper_set)+1):
        for subset in combinations(helper_set, r):
            P = Permutation(N-1)
            for move in subset:
                P *= move
            solved_states.add(P)

    return solved_states

# generators: a set of moves
# N: order of the symmetric group
# k: the current index
# compute_schreiers_vector(): computes Schreier's vector
def compute_schreiers_vector(generators, N, k):
    vector = np.full(N, None)
    vector[k] = Permutation(N-1)
    counter = 1
    while counter > 0:
        counter = 0
        for i in range(N):
            if vector[i] is not None:
                for generator in generators:
                    j = generator(i)
                    if vector[j] is None:
                        vector[j] = vector[i] * generator
                        counter += 1
                    j = (generator**-1)(i)
                    if vector[j] is None:
                        vector[j] = vector[i] * (generator**-1)
                        counter += 1
    return vector

# generators: a set of generators
# schreiers_vector: Schreier's vector of the current stabilizer group
# schreiers_vector_non_zero: all non-zero entries in Schreier's vector
# k: the current index
# generating_set(): computes a generating set for the next stabilizer group
def generating_set(generators, schreiers_vector, schreiers_vector_non_zero, k):
    generating_set = set()
    for a in generators:
        for u in schreiers_vector_non_zero:
            au = u * a
            phi = schreiers_vector[au(k)]
            new_generator = au * phi**-1
            generating_set.add(new_generator)
    return generating_set

# move: a single move
# pair(): finds first index i where move[i] != i (used in Sims filter)
def pair(move):
    i = 0
    while (move(i) == i):
        i += 1
    return i, move(i)

# generators: a set of generators
# N: order of the symmetric group
# sims_filter(): restricts the number of generators to a maximum of N(N-1)/2
def sims_filter(generators, N):
    table = np.full((N, N), None)
    for generator in generators:
        move = generator
        while True:
            if move == Permutation(N-1):
                break
            i, j = pair(move)
            entry = table[i, j]
            if entry is None:
                table[i, j] = move
                break
            else:
                move = entry * move**-1
    return {move for move in table.flatten() if move is not None}