# Sick of stupid subclass.  Combining tetris.py and sim.py
import numpy as np
from matplotlib import pyplot as plt
from meshsolve import solve
from time import time
import matplotlib.gridspec as gridspec
from copy import deepcopy

# TODO:
# Store entire history and make movies quickly

class tetresist():
    def __init__(self, h=20, w=20, R1=10000, R2=10):
        self.R1 = float(R1)
        self.R2 = float(R2)
        self.h = h
        self.w = w
        self.spawnpoint = (-4, w/2 - 1)
        self.tetras = []
        self.gameover = False
        self.history = []
        self.log = []

    def __repr__(self):
        return '{}x{} Tetris instance with {} tetras'.format(self.h, self.w, len(self.tetras))

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
        if rotation is None:
            rotation = np.random.randint(4)

        spawned = tetra(kind, loc)
        spawned.rotate(rotation)
        self.tetras.append(spawned)
        if self.impacto():
            # Spawned on top of something
            self.tetras.pop()
            self.gameover = True
            self.log.append('Failed to spawn at loc {}'.format(loc))
            return 0

        self.history.append(('spawn', {'kind':kind, 'rotation':rotation, 'loc':loc}))
        self.log.append('spawn kind {}, rotation {}, loc {}'.format(kind, rotation, loc))
        return 1

    def delete(self, tetranum=-1):
        del self.tetras[tetranum]
        self.history.append(('delete', {'tetranum':tetranum}))
        self.log.append('delete tetranum {}'.format(tetranum))

    def move(self, x, y, tetranum=-1):
        ''' Try to move a tetra
        self.gameover set if tetra fails to move down and occupies space above 0
        tetra deleted if it moves above x = -4
        Return:
        [n]  : hit tetra n
        [-1] : hit wall
        0    : moved successfully
        These are stupid return codes but I was too lazy to think of better ones
        '''
        piece = self.tetras[tetranum]
        piece.move(x, y)
        impact = self.impacto(tetranum=tetranum)
        if impact:
            # Undo movement
            piece.move(-x, -y)
            self.log.append('failed to move tetra {} x,y = {},{}, hit {}'.format(tetranum, x, y, impact[0]))
            if x != 0:
                # Vertical movement
                if any([o[0] < 0 for o in piece.occupied]):
                    # Piece stops above top
                    self.gameover = True
                #if impact[0] == -1 and x < 0:
                    # Piece goes above -4
                if all([o[0] < 0 for o in piece.occupied]):
                    # Piece is not visible
                    self.delete(tetranum)
            return impact
        # Moved successfully
        self.history.append(('move', {'x':x, 'y':y, 'tetranum':tetranum}))
        self.log.append('move tetra {} x,y = {},{}'.format(tetranum, x, y))
        return 0

    def rotate(self, n=1, tetranum=-1):
        ''' Try to rotate a tetra '''
        piece = self.tetras[tetranum]
        piece.rotate(n)
        if self.impacto(tetranum=tetranum):
            # Failed to rotate, so rotate back
            piece.rotate(4 - n % 4)
            return 0

        self.history.append(('rotate', {'n':n, 'tetranum':tetranum}))
        self.log.append('rotate tetra {}'.format(tetranum))
        return 1

    def impacto(self, tetranum=-1):
        ''' determine if a tetra overlaps another or the wall
        Return
        0 : no impact
        [-1] : hit wall
        [n] : hit tetra n
        if impacts several tetras, only reports first one detected
        '''
        tetra = self.tetras[tetranum]
        # Hit a wall?
        if not all([0 <= o[1] < self.w for o in tetra.occupied]):
            return [-1]
        if not all([-4 <= o[0] < self.h for o in tetra.occupied]):
            return [-1]
        # Hit another piece? Check only ones nearby.
        for i, t in enumerate(self.tetras):
            if i != tetranum % len(self.tetras):
                x1, y1 = tetra.loc
                x2, y2 = t.loc
                if abs(x1 - x2) < 4 and abs(y1 - y2) < 4:
                    if any(tetra.occupied & t.occupied):
                        return [i]
        return 0

    def randgame(self, p=1./3):
        ''' play a game randomly until spawn fails '''
        while not self.gameover:
            self.spawn()
            self.randdrop(p=p)
        return self

    def randdrop(self, tetranum=-1, p=1./3):
        ''' drop one piece randomly '''
        while self.move(1, 0, tetranum) == 0:
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

    def compute(self, V_contact=1):
        t0 = time()
        computed = solve(self.resist(), V_contact)

        # These are all PER VOLT
        self.V = computed['V']
        self.R = computed['R']
        self.I = computed['I']
        self.Ix = computed['Ix']
        self.Iy = computed['Iy']
        self.I_mag = computed['I_mag']
        self.P = computed['P']
        self.Emag = computed['E']
        self.Ex = computed['Ex']
        self.Ey = computed['Ey']

        dt = time() - t0
        self.computetime = dt
        #print('Done.  R = {}. {} seconds'.format(self.R, dt))

    def plot(self, hue=None, ax=None, hl=None, **kwargs):
        ''' Plot current state of the field '''
        if hue is None:
            image = [[(1, 1, 1) for j in range(self.w)] for i in range(self.h)]
            for tet in self.tetras:
                for o in tet.occupied:
                    if o[0] >= 0:
                        image[o[0]][o[1]] = tet.color
        else:
            image = np.nan * np.ones((self.h, self.w))
            #if type(cmap) is str:
                #cmap = plt.cm.get_cmap(cmap)
            for tet in self.tetras:
                occ = list(tet.occupied)
                # Find average value of hue (should be mxn)
                hue_avg = np.sum(hue[zip(*occ)])/len(occ)
                for o in occ:
                    if o[0] >= 0:
                        image[o[0]][o[1]] = hue_avg

        if ax is None:
            ax = plt.gca()
        ax.imshow(image, interpolation='nearest', **kwargs)
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.draw()

        return ax

    def plot_all(self, hue=None, ax=None, **kwargs):
        ''' Plot current state of the field, including spawn area '''
        if hue is None:
            image = [[(1, 1, 1) for j in range(self.w)] for i in range(self.h + 4)]
            for tet in self.tetras:
                for o in tet.occupied:
                    image[o[0]+4][o[1]] = tet.color
        else:
            image = np.nan * np.ones((self.h, self.w))
            #if type(cmap) is str:
                #cmap = plt.cm.get_cmap(cmap)
            for tet in self.tetras:
                occ = list(tet.occupied)
                # Find average value of hue (should be mxn)
                hue_avg = np.sum(hue[zip(*occ)])/len(occ)
                for o in occ:
                    image[o[0]+4][o[1]] = hue_avg

        if ax is None:
            ax = plt.gca()
        ax.imshow(image, interpolation='nearest', **kwargs)
        #ax.hlines(3.5, 0, self.w, linestyles='dashed')
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.draw()

        return ax

    def plotP(self, ax=None, log=False, cmap='hot', interpolation='none', **kwargs):
        ''' plot the power '''
        if ax is None:
            ax = plt.gca()
        if log:
            ax.imshow(np.log(self.P), cmap=cmap, interpolation=interpolation, **kwargs)
        else:
            ax.imshow(self.P, cmap=cmap, interpolation=interpolation, **kwargs)

    def plotI(self, ax=None, cmap='hot', interpolation='none', **kwargs):
        ''' plot the current '''
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.I_mag, cmap=cmap, interpolation=interpolation, **kwargs)

    def plotV(self, ax=None, cmap='hot', interpolation='none', **kwargs):
        ''' plot the voltage '''
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.V, cmap=cmap, interpolation=interpolation, **kwargs)

    def plotI_vect(self, ax=None, **kwargs):
        ''' plot vector field of current'''
        if ax is None:
            ax = plt.gca()
        X, Y = np.meshgrid(range(self.h), range(self.w), indexing='ij')
        ax.streamplot(Y, X, self.Iy, self.Ix, **kwargs)

    def plotE_vect(self, ax=None, **kwargs):
        ''' plot vector field of E'''
        if ax is None:
            ax = plt.gca()
        X, Y = np.meshgrid(range(self.h), range(self.w), indexing='ij')
        ax.streamplot(Y, X, self.Ey, self.Ex, **kwargs)

    def plotE(self, ax=None, cmap='hot', interpolation=None, **kwargs):
        ''' plot the electric field magnitude'''
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.Emag, cmap=cmap, interpolation=interpolation, **kwargs)

    def cool_plot(self, **kwargs):
        fig, ax = plt.subplots()
        self.plot()
        self.plotE_vect()
        self.plotV(alpha=.4, interpolation=None)

    def resist(self):
        field = self.field()
        return (field == 0) * self.R1 + (field * self.R2)


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
        if kind == 'single':
            self.color = (0, 0, 0)
            self.mat = np.array([1])
        else:
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


class rerun(tetresist):
    ''' Use history of another instance to rerun simulation '''
    def __init__(self, parent, start=0):
        tetresist.__init__(self, h=parent.h, w=parent.w, R1=parent.R1, R2=parent.R2)
        self.commands = parent.history
        self.frame = 0
        self.next(start)

    def next(self, numframes=1):
        # Execute numframes of history
        for cmd, args in self.commands[self.frame:self.frame + numframes]:
            getattr(self, cmd)(**args)
        self.frame += numframes

    def prev(self):
        # Does nothing.
        pass

import matplotlib.animation as animation
def movie(game, interval=0.1, skipframes=0, start=0):
    ''' TODO: generate everything before animation somehow '''
    t = rerun(game, start=start)
    def data_gen():
        while t.frame < len(t.commands):
            yield t
            t.next(1+skipframes)

    def run(data):
        try:
            del ax.images[0]
        except:
            pass
        data.plot_all(ax=ax)
        #data.compute()
        #data.plotV(alpha=0.3)
        return ax.images

    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=150, metadata=dict(artist='Me'), bitrate=3000)
    fig, ax = plt.subplots()
    #t.plot(ax=ax)
    #ax.hlines(3.5, -0.5, game.w, linestyles='dashed', zorder=10)
    #spawn = np.zeros((game.h, game.w))
    #spawn[0:4, :] = 1
    #ax.imshow(spawn, alpha=.3, cmap='gray_r', interpolation='none')
    #plt.pause(.3)

    ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=interval, save_count=5000)
    #ani.save('movie.mp4', writer=writer)
    return ani

def movie2(game, interval=0.1, skipframes=0):
    ''' TODO: generate everything before animation somehow '''
    def data_gen():
        t = rerun(game)
        while t.frame < len(t.commands):
            t.compute()
            yield t
            t.next(1+skipframes)

    def run(data):
        try:
            del ax.images[0]
        except:
            pass
        data.plot(hue=data.I_mag, cmap='Reds', ax=ax)
        #data.compute()
        #data.plotV(alpha=0.3)
        return ax.images

    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=150, metadata=dict(artist='Me'), bitrate=3000)
    fig, ax = plt.subplots()

    ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=interval, save_count=5000)
    #ani.save('movie.mp4', writer=writer)
    return ani
