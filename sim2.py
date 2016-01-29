from tetris2 import *
from time import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy
from meshsolve import solve

class tetresist(tetris):
    ''' tetris subclass that knows its voltages and currents '''
    def __init__(self, h=20, w=20, R1=10000, R2=10):
        tetris.__init__(self, h=h, w=w)
        self.V = mesh.newmesh(h, w)
        self.R1 = R1
        self.R2 = R2

    def compute(self):
        t0 = time()
        self.resist = resist(self.field(), self.R1, self.R2)
        self.solution = solve(self.resist)
        # These are all PER VOLT
        self.V = self.solution['V']
        self.R = self.solution['R']
        self.I = self.solution['I']
        self.I_vect = self.solution['I_vect']
        self.I_mag = self.solution['I_mag']
        self.P = self.solution['P']
        dt = time() - t0
        self.computetime = dt
        print('R = {}. {} seconds'.format(self.R, dt))

    def plotP(self, log=False):
        ''' plot the power '''
        if log:
            plt.imshow(np.log(self.P), cmap='hot', interpolation='nearest')
        else:
            plt.imshow(self.P, cmap='hot', interpolation='nearest')

    def plotI(self):
        ''' plot the current '''
        plt.imshow(self.I, cmap='hot', interpolation='nearest')

    def plotV(self):
        ''' plot the voltage '''
        img = [[node.voltage for node in row] for row in self.V]
        plt.imshow(img, interpolation='nearest')

    def plotI_vect(self):
        ''' plot vector field of current on top of blocks '''
        # There's a special kind of plot for this so you might not have to scale
        # in the idiotic way that follows
        self.plot()
        Ix = [[ix/(ix**2 + iy**2)**0.45 for ix, iy in e] for e in self.I_vect]
        # don't know why y is negative.
        Iy = [[-iy/(ix**2 + iy**2)**0.45 for ix, iy in e] for e in self.I_vect]
        plt.quiver(Iy, Ix)

def resist(tetfield, R1=10000., R2=10.):
    ''' Get array of resistances '''
    R1 = float(R1)
    R2 = float(R2)
    return (tetfield == 0) * R1 + (tetfield * R2)

def repeat(h=20, w=10, p=1/3., nsamples=100):
    samples = []
    for i in range(nsamples):
        t = tetresist(h, w)
        t.randgame(p=p)
        t.compute()
        samples.append(t)
        print('{} / {}'.format(i+1, nsamples))

    # Do a bunch of plots
    plt.subplots(4,4)
    for i,t in enumerate(np.random.choice(samples, 16)):
        plt.subplot(4,4,i)
        t.plot()

    return samples


def repro(n=5, h=20, w=10, p=1./3):
    # switching curves reproducibility
    listout = []
    for i in range(n):
        listout.append(switchcurve(h,w,p=p))
        print('Repeat.')

    R = [[t.R for t in repeat] for repeat in switch]

    plt.figure()
    for r in R:
        plt.plot(r)

    return listout

def formingcurve(h, w, every=1, p=1./3):
    samples = []
    t = tetresist(h, w)
    while 1:
        t.compute()
        samples.append(deepcopy(t))
        if t.gameover:
            break
        for _ in range(every):
            t.spawn()
            t.randdrop(p=p)

    #plot it
    plotformingcurve(samples)
    return samples

def plotformingcurve(listoftets):
    Vsub = np.linspace(0, listoftets[1].V_contact, 100)
    # initial current proportional to initial resistance
    Isub = Vsub / listoftets[0].R
    Vcalc = [s.V_contact for s in listoftets[1:]]
    Icalc = [s.I_avg*s.V_contact for s in listoftets[1:]]
    V = np.append(Vsub, Vcalc)
    I = np.append(Isub, Icalc)
    #plt.figure()
    plt.plot(V, I)
    plt.yscale('log')

def cycle(h, w, p=1./3, n=100):
    ''' switch the cell back and forth by deleting the top pieces and dropping new ones '''
    t = tetresist(h, w)
    output = []
    for i in range(n):
        t.randgame(p=p)
        t.compute()
        output.append(deepcopy(t))
        # delete some top piece(s)
        t.rupture(0.5)
        t.compute()
        output.append(deepcopy(t))
        print('{} / {}'.format(i+1, n))

    plt.plot([t.R for t in output[0::2]], '.', label='LRS')
    plt.plot([t.R for t in output[1::2]], '.', label='HRS')
    plt.xlabel('Cycle #')
    plt.ylabel('Resistance (Ohm)')
    plt.legend()
    plt.yscale('log')

    return output

def big_rupture_cycle():
    t = tetresist(100,80)
    figure()
    t.randgame(p=0.2)
    t.plot()
    for _ in range(500):
        #t.plot()
        #plt.pause(.05)
        t.rupture(4)
        #t.plot()
        #plt.pause(.05)
        t.randgame(p=0.5)
    figure()
    t.plot()

def plotsamples(listoftetresists, w=4, h=4, method='plot', slice='rand'):
    ''' Plot from a random sample from the list '''
    if slice == 'rand':
        samples = np.random.choice(listoftetresists, w*h)
    else:
        samples = listoftetresists[slice]
    plt.figure(figsize = (6,6))
    gs1 = gridspec.GridSpec(h, w)
    gs1.update(wspace=0.025, hspace=0.05)

    for i,t in enumerate(samples):
        ax = plt.subplot(gs1[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        getattr(t, method)()

