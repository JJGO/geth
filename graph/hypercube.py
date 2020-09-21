import itertools


class Hypercube:
    def __init__(self, dim):
        self.dim = dim
        self.world_size = 2 ** dim

    def nfaces(self, dim):
        assert dim <= self.dim
        k = self.dim - dim
        Idx = list(reversed(range(self.dim)))
        faces = []
        for idx in itertools.combinations(Idx, k):
            for vals in itertools.product((0, 1), repeat=k):
                #                 print(idx, vals)
                common = 0
                for i, v in zip(idx, vals):
                    common ^= v << i
                face = []
                remaining_idx = [i for i in Idx if i not in idx]
                for remaining_vals in itertools.product((0, 1), repeat=dim):
                    vertex = common
                    for i, v in zip(remaining_idx, remaining_vals):
                        vertex ^= v << i
                    face.append(vertex)
                faces.append(face)
        return faces

    def vertices(self):
        return self.nfaces(0)

    def edges(self):
        return self.nfaces(1)

    def faces(self):
        return self.nfaces(2)

    def cells(self):
        return self.nfaces(3)
