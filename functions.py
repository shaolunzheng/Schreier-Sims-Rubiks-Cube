import numpy as np
from sympy.combinatorics import Permutation
from itertools import combinations
import config

# n: size of the Rubik's cube (n x n x n)
# N: order of the symmetric group
# enumerate(): enumerates the moves that generate the Rubik's cube
def enumerate_cube(n, N):
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
# get_human_order(): mimics a human solving technique by choosing a specific order of stabilization
# still needs to be automated
def get_human_order(n):
    if n == 3:
        return [(1,),(3,),(5,),(7,),(0,),(2,),(4,),(6,),(9,),(13,),(35,),(39,),(25,),(27,),(29,),(31,),(24,),(26,),(28,),(30,)]
    if n == 4:
        return [3,7,11,15,51,55,59,63,19,23,27,31,35,39,43,47,67,71,75,79,83,87,91,95,
                1,6,5,10,9,14,13,2,0,4,8,12,17,22,25,30,66,77,69,74,49,54,53,58,57,62,61,50,48,52,56,60,
                20,21,26,24,36,37,42,40,64,65,70,68,80,81,86,84,16,18,29,28,32,34,45,44,72,73,78,76,88,89,94,92,33,38,41,46,82,93,85,90]
# get_human_order_multiple(): stabilizes multiple elements at once
def get_human_order_multiple(n):
    if n == 3:
        return [(1,3,5,7),(0,2,4,6),(9,13,35,39),(25,27,29,31),(24,26,28,30)]

# n: size of the Rubik's cube (n x n x n)
# N: order of the symmetric group
# get_solved_states(): computes the set of visually solved states for a Rubik's cube of size n
def get_solved_states(n, N):
    result = set()
    result.add(Permutation(N-1))
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
            result.add(P)

    return result

# move: a single move
# value: length of the move written as composition of original generators
# update_length(): determines whether a shorter length already exists, then assigns the shortest length
def update_length(move, value):
    if value < config.lengths.get(move, float('inf')):
        config.lengths[move] = value

# generators: a set of moves
# N: order of the symmetric group
# k: the current index list
# compute_schreiers_tree(): computes Schreier's tree (a dictionary)
def compute_schreiers_tree(generators, N, k):
    d = {k: Permutation(N - 1)}
    counter = 1
    while counter > 0:
        counter = 0
        keys = list(d.keys())
        for i in keys:
            for generator in generators:
                j = tuple(generator(x) for x in i)
                if j not in d.keys():
                    d[j] = d[i] * generator
                    update_length(d[j], config.lengths[d[i]] + config.lengths[generator])
                    counter += 1
                generator_inverse = generator**-1
                j = tuple(generator_inverse(x) for x in i)
                if j not in d.keys():
                    d[j] = d[i] * generator_inverse
                    update_length(d[j], config.lengths[d[i]] + config.lengths[generator])
                    counter += 1
    return d

# generators: a set of generators
# schreiers_tree: Schreier's tree of the current stabilizer group (a dictionary)
# k: the current index list
# compute_generating_set(): computes a generating set for the next stabilizer group
def compute_generating_set(generators, schreiers_tree, k):
    result_set = set()
    for a in generators:
        for u in schreiers_tree.values():
            au = u * a
            au_k = tuple(au(x) for x in k)
            pi = schreiers_tree[au_k]
            new_generator = au * pi**-1
            update_length(new_generator, config.lengths[a] + config.lengths[u] + config.lengths[pi])
            result_set.add(new_generator)
    return result_set

# move: a single move
# pair(): finds first index i, where move[i] != i (used in Sims filter)
def pair(move):
    i = 0
    while move(i) == i:
        i += 1
    return i, move(i)

# generators: a set of generators
# N: order of the symmetric group
# apply_sims_filter(): restricts the number of generators to a maximum of N(N-1)/2
def apply_sims_filter(generators, N):
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
                new_move = entry * move**-1
                update_length(new_move, config.lengths[entry] + config.lengths[move])
                move = new_move
    return {move for move in table.flatten() if move is not None}