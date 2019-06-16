import itertools

import numpy

def full_3x3_to_Voigt_6_index(i, j):
    if i == j:
        return i
    return int(6-i-j)

def Voigt_6x6_to_full_3x3x3x3_storage_indices(C_6x6,C_3x3x3x3):
    
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        Voigt_i = full_3x3_to_Voigt_6_index(i, j)
        Voigt_j = full_3x3_to_Voigt_6_index(k, l)
        C_3x3x3x3[i, j, k, l,0] = C_6x6[Voigt_i, Voigt_j]
    return C_3x3x3x3

def Get_C_6x6_storage():
    return numpy.array([
        [[0],[1] ,[2] ,[3] ,[4] ,[5] ],
        [[1],[6] ,[7] ,[8] ,[9] ,[10]],
        [[2],[7] ,[11],[12],[13],[14]],
        [[3],[8] ,[12],[15],[16],[17]],
        [[4],[9] ,[13],[16],[18],[19]],
        [[5],[10],[14],[17],[19],[20]]
        ],dtype=numpy.int64)

def Get_C_3x3x3x3_storage():
    C_3x3x3x3_storage = numpy.zeros((3,3,3,3,1),dtype=numpy.int64)
    C_3x3x3x3_storage = Voigt_6x6_to_full_3x3x3x3_storage_indices(
        Get_C_6x6_storage(),
        C_3x3x3x3_storage
    )
    return C_3x3x3x3_storage