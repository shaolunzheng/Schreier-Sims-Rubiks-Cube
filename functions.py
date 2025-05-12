import numpy as np

# moves: an np.array containing multiple moves
# n: order of symmetric group
# compose: composes the elements in moves in order
def compose(moves, n):
    result = np.arange(n)
    for move in moves:
        result = move[result]
    return result

# move: a single move
# inverse: computes the inverse of this move
def inverse(move, n):
    result = np.arange(n)
    result[move] = result
    return result

# generators: an np.array of moves
# k: the current index
# computes Schreiers vector
def compute_schreiers_vector(generators, n, k):
    result = [np.zeros(n, dtype = int) for _ in range(n)]
    result[k] = np.arange(n)
    v = [False for _ in range(n)]
    v[k] = True
    counter = 1
    while counter > 0:
        counter = 0
        for i in range(n):
            if v[i]:
                for generator in generators:
                    j = generator[i]
                    if not v[j]:
                        v[j] = True
                        result[j] = compose([result[i], generator], n)
                        counter += 1
                    j = inverse(generator, n)[i]
                    if not v[j]:
                        v[j] = True
                        result[j] = compose([result[i], inverse(generator, n)], n)
                        counter += 1
    return result

# generators: an np.array of generators
# schreiers_vector: Schreier's vector of the current stabilizer group
# schreiers_vector_non_zero: all non-zero entries in Schreier's vector
# k: the current index
# computes a generating set for the next stabilizer group
def generating_set(generators, schreiers_vector, schreiers_vector_non_zero, n, k):
    result = []
    # use helper set to efficiently check membership (np.array doesn't support hashing)
    helper_set = set()
    for a in generators:
        for u in schreiers_vector_non_zero:
            au = compose([u, a], n)
            phi = schreiers_vector[au[k]]
            new_generator = compose([au, inverse(phi, n)], n)
            if tuple(new_generator) not in helper_set:
                helper_set.add(tuple(new_generator))
                result.append(new_generator)
    return result

# move: a single move
# finds first index i where move[i] != i (used in Sims filter)
def pair(move):
    i = 0
    while (move[i] == i):
        i += 1
    return i, move[i]

# generators: a np.array of generators
# restricts the number of generators to a maximum of n(n-1)/2
def sims_filter(generators, n):
    table = np.zeros((n, n, n), dtype = int)
    k = 0
    while (k < len(generators)):
        move = generators[k]
        if np.array_equal(move, np.arange(n)):
            k += 1
        else:
            i, j = pair(move)
            entry = table[i,j]
            if not np.any(entry != 0):
                table[i,j] = move
                k += 1
            else:
                generators[k] = compose([entry, inverse(move, n)], n)

    helper_table = np.any(table != 0, axis = 2)
    result = [generator for generator in table[helper_table]]

    return result