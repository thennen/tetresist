import numpy as np
from matplotlib import pyplot as plt

class tetris():
    ''' Play tetris very poorly '''
    def __init__(self, h=20, w=10):
        self.h = h
        self.w = w
        self.spawnpoint = w/2 - 1
        self._field = np.zeros((h,w))
        self.border = np.ones((self.h+4+3, self.w+2*3))
        self.border[0:4,:] = 0
        self.border[4:-3, 3:-3] = self._field
        self.tetras = []
        self.gameover = False
        self.movehistory = []

    @property
    def field(self):
        return self._field

    @field.getter
    def field(self):
        fh, fw = self._field.shape
        field = [[0 for j in range(fw)] for i in range(fh+4)]
        for tet in self.tetras:
            x, y = tet.loc
            dx, dy = tet.mat.shape
            sp = self.spawnpoint
            for i in range(dx):
                for j in range(dy):
                    if tet.mat[i,j]:
                        field[i+x][j+sp+y] = 1
        # cut off spawn area
        field = field[4:]
        return np.array(field)


    def spawn(self, kind=None, rotation=None):
        ''' generate a random tetra with random rotation '''
        if kind is None:
            kind = np.random.choice(tetra.kinds)
            spawned = tetra(kind)
            if rotation is None:
                spawned.rotate(np.random.randint(4))
            else:
                spawned.rotate(rotation)
        else:
            spawned = tetra(kind)
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
        dloc = np.array([x, y])
        piece = self.tetras[tetranum]
        piece.loc += dloc
        if self.impacto():
            piece.loc -= dloc
            if x > 0:
                if piece.loc[0] + piece.top <= 4:
                    # Piece touches top row
                    self.gameover = True
                return 0
            else:
                # Impact on the side
                return 2
        self.movehistory.append((x, y, tetranum % len(self.tetras)) )

        return 1

    def impacto(self, tetranum=-1):
        ''' determine if a tetra overlaps another or the wall '''
        tetra = self.tetras[tetranum]
        tx, ty = tetra.loc
        dx, dy = tetra.mat.shape

        # see if it hit the wall
        sp = self.spawnpoint
        bx, by = tx, ty + sp + 3
        if np.any(self.border[bx:bx+dx, by:by+dy] * tetra.mat):
            return 1

        # check if any near ones touch
        overlap = np.zeros((13, 13))
        overlap[6:6+dx, 6:6+dy] = tetra.mat
        for i,t in enumerate(self.tetras):
            if i != tetranum % len(self.tetras):
                x, y = t.loc
                dx2, dy2 = t.mat.shape
                if abs(tx - x) < 4 and abs(ty - y) < 4:
                    overlap[6+x-tx:6+x-tx+dx2, 6+y-ty:6+y-ty+dy2] += t.mat
                    if np.any(overlap > 1):
                        # Impacto with tetra i
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
        while self.move(1,0, tetranum):
            trans = np.random.choice((-1,0,1), p=(p,1 - 2*p, p))
            self.move(0,trans)
        return self

    def drop(self, tetranum=-1):
        ''' drop one piece straight down '''
        while self.move(1, 0, tetranum):
            pass
        return self

    def randmove(self, tetranum=-1, p=1/3., x=1):
        trans = np.random.choice((-1,0,1), p=(p,1 - 2*p, p))
        self.move(0, trans, tetranum)
        return self.move(x, 0, tetranum)


    def plot(self, ax=None):
        ''' Plot current state of the field '''
        fh, fw = self._field.shape
        #image = np.array(np.zeros((fw +4, fh)))
        image = [[(1,1,1) for j in range(fw)] for i in range(fh+4)]
        for tet in self.tetras:
            x, y = tet.loc
            dx, dy = tet.mat.shape
            sp = self.spawnpoint
            for i in range(dx):
                for j in range(dy):
                    if tet.mat[i,j]:
                        image[i+x][j+sp+y]= tet.color
        # cut off spawn area
        image = image[4:]
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

    def __init__(self, kind):
        self.color = tetra.colors[kind]
        self.mat = tetra.mats[kind]
        self.loc = np.array([0,0])
        self.rot = 0
        self.kind = kind
        firstrow = [any(m) for m in self.mat]
        self.top = firstrow.index(True)

    def rotate(self, n=1):
        ''' Rotate 90 deg n times'''
        for _ in range(n):
            self.mat = np.rot90(self.mat)
        self.rot = (self.rot + n) % 4
        firstrow = [any(m) for m in self.mat]
        self.top = firstrow.index(True)
        return self

