import numpy as np

class Block:
    def __init__(self, block_id=0, rotation=0):
        self.block_id = block_id
        self.rotation = rotation

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(Block.SHAPES[self.block_id])

    def resetRotation(self):
        self.rotation = 0

    def getCoords(self):
        shape = Block.SHAPES[self.block_id][self.rotation]
        r, c = shape.shape
        for x in range(r):
            for y in range(c):
                if shape[x][y] == 1:
                    yield x, y

    def _shapeStr(self):
        shape = Block.SHAPES[self.block_id][self.rotation]
        r, c = shape.shape
        res = ''
        for x in range(r):
            for y in range(c):
                if shape[x][y] == 1:
                    res += '■ '
                else:
                    res += '. '
            res += '\n'
        return res

    def __str__(self):
        return f'{Block.SHAPE_NAMES[self.block_id]}-{self.rotation}\n{self._shapeStr()}'
    
    # I, O, T, S, Z, J, L
    # 0, 1, 2, 3, 4, 5, 6

    SHAPE_NAMES = ['I Block', 'O Block', 'T Block', 'S Block', 'Z Block', 'J Block', 'L Block']

    i_block = [ 
        # rotation 0
        np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]),
        # rotation 1
        np.array([
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ])
    ]

    o_block = [
        # rotation 0
        np.array([
            [1, 1],
            [1, 1],
        ])
    ]

    t_block = [
        # rotation 0
        np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]),
        # rotation 1
        np.array([
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]),
        # rotation 2
        np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0]
        ]),
        # rotation 3
        np.array([
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 0]
        ])
    ]

    s_block = [
        # rotation 0
        np.array([
            [0, 0, 0],
            [0, 1, 1],
            [1, 1, 0]
        ]),
        # rotation 1
        np.array([
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1]
        ])
    ]

    z_block = [
        # rotation 0
        np.array([
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
        ]),
        # rotation 1
        np.array([
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0]
        ])
    ]

    j_block = [
        # rotation 0
        np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 1]
        ]),
        # rotation 1
        np.array([
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 0]
        ]),
        # rotation 2
        np.array([
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ]),
        # rotation 3
        np.array([
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 0]
        ])
    ]

    l_block = [
        # rotation 0
        np.array([
            [0, 0, 0],
            [1, 1, 1], 
            [1, 0, 0]
        ]),
        # rotation 1
        np.array([
            [1, 1, 0], 
            [0, 1, 0],
            [0, 1, 0]
        ]),
        # rotation 2
        np.array([
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 0]
        ]),
        # rotation 3
        np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1]
        ])
    ]

    SHAPES = [i_block, o_block, t_block, s_block, z_block, j_block, l_block]
    NUM_SHAPES = len(SHAPES)

