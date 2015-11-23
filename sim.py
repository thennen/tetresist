from tetris import *
from meshsolve import solve
from time import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy

class tetresist(tetris):
    ''' tetris subclass that knows its voltages and currents '''
    def __init__(self, h=20, w=20, R1=10000, R2=10):
        tetris.__init__(self, h=h, w=w)
        self.R1 = float(R1)
        self.R2 = float(R2)

    def compute(self, V_contact=1):
        t0 = time()
        # Approximate V_contact from number of pieces on the board
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
        print('Done.  R = {}. {} seconds'.format(self.R, dt))

    def plotP(self, log=False, cmap='hot', interpolation='none', **kwargs):
        ''' plot the power '''
        if log:
            plt.imshow(np.log(self.P), cmap=cmap, interpolation=interpolation, **kwargs)
        else:
            plt.imshow(self.P, cmap=cmap, interpolation=interpolation, **kwargs)

    def plotI(self, cmap='hot', interpolation='none', **kwargs):
        ''' plot the current '''
        plt.imshow(self.I_mag, cmap=cmap, interpolation=interpolation, **kwargs)

    def plotV(self, cmap='hot', interpolation='none', **kwargs):
        ''' plot the voltage '''
        plt.imshow(self.V, cmap=cmap, interpolation=interpolation, **kwargs)

    def plotI_vect(self, **kwargs):
        ''' plot vector field of current'''
        X, Y = np.meshgrid(range(self.h), range(self.w), indexing='ij')
        plt.streamplot(Y, X, self.Iy, self.Ix, **kwargs)

    def plotE_vect(self, **kwargs):
        ''' plot vector field of E'''
        X, Y = np.meshgrid(range(self.h), range(self.w), indexing='ij')
        plt.streamplot(Y, X, self.Ey, self.Ex, **kwargs)

    def cool_plot(self, **kwargs):
        fig, ax = plt.subplots()
        self.plot()
        self.plotE_vect()
        self.plotV(alpha=.4, interpolation=None)

    def resist(self):
        field = self.field()
        return (field == 0) * self.R1 + (field * self.R2)

