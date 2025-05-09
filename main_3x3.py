import numpy as np
from functions import compose, inverse, compute_schreiers_vector, generating_set, sims_filter

turns = {
    "E": np.array(list(range(48))),
    "R": np.array([28,3,4,5,6,7,8,1,2,9,10,11,12,13,14,15,16,17,18,47,0,41,22,23,24,
            25,26,35,36,37,30,31,32,33,34,19,20,21,38,39,40,29,42,43,44,45,46,27]),
    "L": np.array([0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,9,10,33,18,19,20,21,22,39,40,
            45,26,27,28,29,30,43,44,25,34,35,36,37,38,31,32,41,42,23,24,17,46,47]),
    "U": np.array([0,33,34,35,4,5,6,7,8,41,42,43,12,13,14,15,16,19,20,21,22,23,24,17,18,
            25,26,27,28,29,30,31,32,9,10,11,36,37,38,39,40,1,2,3,44,45,46,47]),
    "D": np.array([0,1,2,3,4,45,46,47,8,9,10,11,12,37,38,39,16,17,18,19,20,21,22,23,24,
            27,28,29,30,31,32,25,26,33,34,35,36,5,6,7,40,41,42,43,44,13,14,15]),
    "F": np.array([0,27,2,3,4,5,6,25,26,9,10,21,22,23,14,15,16,17,18,19,20,7,8,1,24,
            11,12,13,28,29,30,31,32,35,36,37,38,39,40,33,34,41,42,43,44,45,46,47]),
    "B": np.array([42,1,2,17,18,19,6,7,8,31,10,11,12,13,14,29,30,15,16,9,20,21,22,23,24,
            25,26,27,28,3,4,5,32,33,34,35,36,37,38,39,40,43,44,45,46,47,0,41])
}

n = 48

# generating set
generators = [turns["R"], turns["L"], turns["U"], turns["D"], turns["F"], turns["B"]]

# order in which elements should be stabilized
solving_order = np.array([26,28,30,32,25,27,29,31,4,8,12,16,17,19,21,18,20,22,
                   1,2,3,5,6,7,9,10,11,13,14,15,23,24,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,0])

# list of group orders of the stabilizer groups
orders_list = []

# list of generating sets of the stabilizer groups
generators_list = []

# list of Schreiers vectors for the stabilizer groups
schreiers_vectors_list = []

# the algorithm

i = 0

temp_generators = generators

while(len(temp_generators) > 0):
    k = solving_order[i]

    generators_list.append(temp_generators)

    schreiers_vector = compute_schreiers_vector(temp_generators, n, k)
    schreiers_vectors_list.append(schreiers_vector)

    schreiers_vector_non_zero = [element for element in schreiers_vector if np.any(element != 0)]

    order_orbit = len(schreiers_vector_non_zero)
    orders_list.append(order_orbit)

    new_generators = generating_set(temp_generators, schreiers_vector, schreiers_vector_non_zero, n, k)
    temp_generators = sims_filter(new_generators, n)

    i += 1

# order of 3x3 rubiks cube group
orders_list = np.array(orders_list, dtype=object)
print(np.prod(orders_list))