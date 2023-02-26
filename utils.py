import json

import matrix

import numpy as np


def read_triagonal_matrix(filename, matrix, vector):
    with open(filename, 'r') as json_data:
        data = json.load(json_data)[0] # !
        matrix.a = [0] + data['A']
        matrix.b = data['B']
        matrix.c = data['C'] + [0]
        vector.data = data['D']


def complex_to_list(list_):
    list_without_complex = []
    for obj in list_:
        if isinstance(obj, (complex, np.complex)):
            list_without_complex.append([obj.real, obj.imag])
        else:
            list_without_complex.append(obj)
    return list_without_complex
