import numpy as np

# moves: a list containing multiple moves
# n: order of symmetric group
# compose: composes the elements in moves in order
def compose(moves, n):
    result = np.array(list(range(n)))
    for move in moves:
        result = move[result]
    return result

# move: a single move
# inverse: computes the inverse of this move
def inverse(move, n):
    result = np.array(list(range(n)))
    result[move] = result
    return result

# computes Schreiers vector
def compute_schreiers_vector(generators, n, k):
    result = [np.zeros(n, dtype = int) for _ in range(n)]
    result[k] = np.array(list(range(n)))
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

# computes a generating set for the next stabilizer group
def generating_set(temp_generators, schreiers_vector, schreiers_vector_non_zero, n, k):
    result = []
    # use helper set to efficiently check membership (np.array doesn't support hashing)
    helper_set = set()
    for a in temp_generators:
        for u in schreiers_vector_non_zero:
            au = compose([u, a], n)
            phi = schreiers_vector[au[k]]
            new_generator = compose([au, inverse(phi, n)], n)
            if tuple(new_generator) not in helper_set:
                helper_set.add(tuple(new_generator))
                result.append(new_generator)
    return result

# finds first index i where move[i] != i (used in Sims filter)
def pair(move):
    i = 0
    while (move[i] == i):
        i += 1
    return i, move[i]

# restricts the number of generators to a maximum of n(n-1)/2
def sims_filter(generators, n):
    table = np.zeros((n, n, n), dtype = int)
    k = 0
    while (k < len(generators)):
        move = generators[k]
        if np.array_equal(move, np.array(list(range(n)))):
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