import numpy as np
from matplotlib import pyplot as plt


class tetris():
    ''' Play tetris very poorly '''
    def __init__(self, h=20, w=10):
        self.h = h
        self.w = w
        self.spawnpoint = (-4, w/2 - 1)
        self.tetras = []
        self.gameover = False
        self.movehistory = []

    def field(self):
        field = np.zeros((self.h, self.w))
        for tet in self.tetras:
            for o in tet.occupied:
                if o[0] >= 0:
                    field[o] = 1
        return field

    def spawn(self, kind=None, rotation=None, loc=None):
        ''' generate a random tetra with random rotation '''
        if loc is None:
            loc = self.spawnpoint
        if kind is None:
            kind = np.random.choice(tetra.kinds)
            spawned = tetra(kind, loc)
            if rotation is None:
                spawned.rotate(np.random.randint(4))
            else:
                spawned.rotate(rotation)
        else:
            spawned = tetra(kind, loc)
            if rotation is None:
                spawned.rotate(np.random.randint(4))
            else:
                spawned.rotate(rotation)
        self.tetras.append(spawned)
        if self.impacto():
            # Spawned on top of something
            self.tetras.pop()
            self.gameover = True
            return 0
        return 1

    def move(self, x, y, tetranum=-1):
        ''' Try to move a tetra '''
        piece = self.tetras[tetranum]
        piece.move(x, y)

        if self.impacto():
            piece.move(-x, -y)
            if x > 0:
                if any([o[0] < 0 for o in piece.occupied]):
                    # Piece touches top row
                    self.gameover = True
                return 0
            else:
                # Impact on the side
                return 2
        self.movehistory.append((x, y, tetranum % len(self.tetras)))

        return 1

    def impacto(self, tetranum=-1):
        ''' determine if a tetra overlaps another or the wall '''
        tetra = self.tetras[tetranum]
        # Hit a wall?
        if not all([0 <= o[1] < self.w for o in tetra.occupied]):
            return 1
        if not all([-4 <= o[0] < self.h for o in tetra.occupied]):
            return 1

        # Hit another piece? Check only ones nearby.
        for i, t in enumerate(self.tetras):
            if i != tetranum % len(self.tetras):
                x1, y1 = tetra.loc
                x2, y2 = t.loc
                if abs(x1 - x2) < 4 and abs(y1 - y2) < 4:
                    if any(tetra.occupied & t.occupied):
                        return 1

        return 0

    def randgame(self, p=1./3):
        ''' play a game randomly until spawn fails '''
        while not self.gameover:
            self.spawn()
            self.randdrop(p=p)
        return self

    def randdrop(self, tetranum=-1, p=1./3):
        ''' drop one piece randomly '''
        while self.move(1, 0, tetranum):
            trans = np.random.choice((-1, 0, 1), p=(p, 1 - 2*p, p))
            self.move(0, trans)
        return self

    def drop(self, tetranum=-1):
        ''' drop one piece straight down '''
        while self.move(1, 0, tetranum):
            pass
        return self

    def randmove(self, tetranum=-1, p=1/3., x=1):
        trans = np.random.choice((-1, 0, 1), p=(p, 1 - 2*p, p))
        self.move(0, trans, tetranum)
        return self.move(x, 0, tetranum)

    def plot(self, ax=None):
        ''' Plot current state of the field '''
        image = [[(1,1,1) for j in range(self.w)] for i in range(self.h)]
        for tet in self.tetras:
            for o in tet.occupied:
                if o[0] >= 0:
                    image[o[0]][o[1]] = tet.color

        if ax is None:
            ax = plt.gca()
        ax.imshow(image, interpolation='nearest')
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.draw()

        return ax

    def video(self):
        ''' Piece by piece video '''
        pass

class tetra():
    ''' class for each tetragon.  Knows its shape, color, and location '''

    mats = [np.array([[0, 0, 0, 0],
                      [1, 1, 1, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]),
            np.array([[1, 0, 0],
                      [1, 1, 1],
                      [0, 0, 0]]),
            np.array([[0, 0, 1],
                      [1, 1, 1],
                      [0, 0, 0]]),
            np.array([[0, 1, 1],
                      [1, 1, 0],
                      [0, 0, 0]]),
            np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 0, 0]]),
            np.array([[1, 1, 0],
                      [0, 1, 1],
                      [0, 0, 0]]),
            np.array([[1,1],
                      [1,1]])]
    colors = [(0,1,1),
              (0,0,1),
              (1, 153./255, 51./255),
              (51./255, 1, 51./255),
              (153./255, 51./255, 1),
              (1, 0, 0),
              (1, 1, 0)]

    kinds = range(len(mats))

    def __init__(self, kind=1, loc=(0,0)):
        self.color = tetra.colors[kind]
        self.mat = tetra.mats[kind]
        self.loc = list(loc)
        self.rot = 0
        self.kind = kind
        self.calc_occupied()

    def move(self, dx, dy):
        self.loc[0] += dx
        self.loc[1] += dy
        self.calc_occupied()

    def calc_occupied(self):
        w = np.where(self.mat)
        self.occupied = set(zip(self.loc[0] + w[0], self.loc[1] + w[1]))

    def rotate(self, n=1):
        ''' Rotate 90 deg n times'''
        for _ in range(n):
            self.mat = np.rot90(self.mat)
        self.rot = (self.rot + n) % 4
        self.calc_occupied()
        return self
