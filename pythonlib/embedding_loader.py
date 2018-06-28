import numpy
import random

def load_w2v(file):
    with open(file) as w2v:
        embeddings = []
        for line in w2v:
            vector = [float(f) for f in line.split(',')]
            embeddings.append(vector)
        vector1 = [random.uniform(0,1.0) for i in range(0, 100)]
        vector2 = [random.uniform(0,1.0) for i in range(0, 100)]
        vector3 = [random.uniform(0,1.0) for i in range(0, 100)]
        vector4 = [random.uniform(0,1.0) for i in range(0, 100)]
        embeddings.append(vector1)
        embeddings.append(vector2)
        embeddings.append(vector3)
        embeddings.append(vector4)

    return numpy.array(embeddings)
