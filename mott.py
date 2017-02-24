'''
Going to copy published Mott material volatile switching simulation through a small modification of my idiot code

todo

keep connection matrix in memory and make modifications on pixel toggles
add reset method

'''
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
# from time import time
import pickle
from scipy.sparse import coo_matrix
import matplotlib.animation as animation
from scipy.sparse.linalg import spsolve
from itertools import groupby
import os


class mott():
    def __init__(self, h=40, w=128, R1=16., R2=0.3):
        self.R1 = float(R1)
        self.R2 = float(R2)
        self.h = h
        self.w = w
        self.log = []
        self.mat = np.zeros((h, w), dtype=bool)
        # List of points that cannot toggle
        self.fixed = []

        # For most recently computed transition rates
        self.gamma_mat = np.zeros((h, w))
        self.E_i_m = []

        self.Rseries = 1.
        self.f0 = 1.
        self.Eb = 20.
        self.Ei = 0.
        self.Em = 10.
        self.beta = 1.
        self.kT = 1

        # This should be the voltage threshold
        self.Vthresh = self.Em / self.beta * self.h

        # Not in stolier paper
        # factor for power -> temperature
        self.alpha = 1
        self.T = np.zeros((h,w))
        self.latticep = 1.

        # Updated every self.step()
        self.time = [0]

        # Updated every compute()
        self.iv = iv()
        self.compute()

        self._bottomrow = np.zeros((h, w), dtype=bool)
        self._bottomrow[-1, :] = True

        self._toprow= np.zeros((h, w), dtype=bool)
        self._toprow[0, :] = True

        self.growdiag = False

    def __repr__(self):
        stringout = ('{}x{} mott instance\n'
                     'Eb0 = {}\n'
                     'Eb1 = {}\n'
                     'beta = {}\n'
                     'alpha = {}\n'
                     'R1 = {}\n'
                     'R2 = {}\n'
                     'f0 = {}\n'
                     'latticep = {}')
        return stringout.format(self.h, self.w, self.Eb0, self.Eb1, self.beta,
                                self.alpha, self.R1, self.R2, self.f0, self.latticep)

    def toggle(self, (i, j)):
        ''' Toggle pixel i, j '''
        if (i,j) not in self.fixed:
            self.mat[i, j] = not self.mat[i, j]
            self.log.append((i, j))
        else:
            self.log.append(np.nan)

    def neighbormask(self):
        ''' Return boolean mask of neighbors '''
        # Could use np.roll if you want to wrap left-right

        pad = np.pad(self.mat, ((1, 1), (1, 1)), mode='constant')
        right = pad[1:-1, :-2]
        left = pad[1:-1, 2:]
        bottom = pad[:-2, 1:-1]
        top = pad[2:, 1:-1]

        # neighbors = (right | left | bottom | top | self._toprow) & ~self.mat

        edges = right | left | top | bottom

        if self.growdiag:
            # Include diagonal neighbors
            bottomleft = pad[:-2, 2:]
            bottomright = pad[:-2, :-2]
            topright = pad[2:, :-2]
            topleft = pad[2:, 2:]

            diag = topleft | topright | bottomright | bottomleft

            neighbors = (edges | diag | self._toprow | self._bottomrow) & ~self.mat
        else:
            neighbors = (edges | self._toprow | self._bottomrow) & ~self.mat

        return neighbors

    def neighbor_masks(self):
        ''' Return boolean mask for empty neighbor pixels, and occupied pixels
        which have neighbors, separate masks for each direction'''
        pad = np.pad(self.mat, ((1, 1), (1, 1)), mode='constant')
        right = pad[1:-1, :-2]
        left = pad[1:-1, 2:]
        bottom = pad[:-2, 1:-1]
        top = pad[2:, 1:-1]

        # u is for unoccupied
        # u_above means a 0 pixel is above a 1 pixel
        u_above = (top | self._bottomrow) & ~self.mat
        u_below = (bottom | self._toprow) & ~self.mat
        u_left = left & ~self.mat
        u_right = right & ~self.mat

        # o is for occupied
        # o_above means a 1 pixel is above a 0 pixel
        o_above = ~top & self.mat & ~self._bottomrow
        o_below = ~bottom & self.mat & ~self._toprow
        o_left = ~left & self.mat
        o_right = ~right & self.mat

        # (up, down, left, right)
        return ((u_above, u_below, u_left, u_right),
                (o_above, o_below, o_left, o_right))

    def gammafunc(self):
        # 1 -> 0 has a different rate function than 0 -> 1
        # Avoid calculating rates for both cases

        # insulator to metal barrier reduced by E
        # 0 -> 1
        insulator_ind = np.where(~self.mat)
        E_i_m = self.Eb - self.beta * self.Emag[insulator_ind]
        self.E_i_m = E_i_m
        gamma_i_m = self.f0 * np.exp(-E_i_m) / self.kT

        # Metal to insulator not effected by E, no need to repeat calculation
        # 1 -> 0
        E_m_i = self.Eb - self.Em
        gamma_m_i = self.f0 * np.exp(-E_m_i) / self.kT

        # Construct and return gamma matrix
        gamma_mat = gamma_m_i * self.mat
        gamma_mat[insulator_ind] = gamma_i_m
        self.gamma_mat = gamma_mat
        return gamma_mat


    def choose_pixel(self):
        ''' Calculate transition rates, return pixel to toggle, and time step'''

        # Simple approach similar to Stolier paper (I think)

        gamma_mat = self.gammafunc()

        # Calculate time step
        gamma_sum = np.sum(gamma_mat)
        dt = - 1 / gamma_sum * np.log(np.random.rand())

        # Pick one of the pixels
        gamma = gamma_mat.ravel()
        p = gamma / gamma_sum
        togglen = np.random.choice(np.arange(len(gamma)), p=p)
        pix = (togglen/self.w, togglen % self.w)

        return pix, dt

    def step(self):
        # Toggle a pixel somewhere
        pix, dt = self.choose_pixel()
        self.toggle(pix)
        self.time.append(self.time[-1] + dt)

    def points(self):
        ''' return boolean mask of point locations '''
        pad = np.pad(self.mat, ((1, 1), (1, 1)), mode='constant')
        right = pad[1:-1, :-2].astype(np.int8)
        left = pad[1:-1, 2:].astype(np.int8)
        bottom = pad[:-2, 1:-1].astype(np.int8)
        top = pad[2:, 1:-1].astype(np.int8)

        pointmask = (self.mat + right + left + bottom + top == 2) & self.mat
        # Bottom row doesn't count
        pointmask = pointmask & ~self._bottomrow

        return pointmask

    def compute(self, V_contact=1, saveiv=False):

        if V_contact != 0:
            computed = solve(self.resist(), V_contact)
            # If there is a series resistor, multiply everything by this scale
            dividerscale = computed['R'] / (computed['R'] + self.Rseries)
            self.V_contact = V_contact * dividerscale
            self.V = computed['V'] * dividerscale
            self.I = computed['I'] * dividerscale
            self.Ix = computed['Ix'] * dividerscale
            self.Iy = computed['Iy'] * dividerscale
            self.I_mag = computed['I_mag'] * dividerscale
            self.P = computed['P'] * dividerscale
            # Probably not right yet
            #self.T = solve_heat(self.P)
            self.Emag = computed['E'] * dividerscale / self.latticep
            self.Ex = computed['Ex'] * dividerscale / self.latticep
            self.Ey = computed['Ey'] * dividerscale / self.latticep
        else:
            # Perfectly reasonable to apply 0 V, but problematic for this algorithm
            # for the purpose of calculating R, pretend V_contact is 1
            computed = solve(self.resist(), 1)
            zeros = np.zeros((self.h, self.w))
            self.V_contact = 0.
            self.V = zeros
            self.I = 0.
            self.Ix = zeros
            self.Iy = zeros
            self.I_mag = zeros
            self.P = zeros
            self.Emag = zeros
            self.Ex = zeros
            self.Ey = zeros

        self.R = computed['R']

        if saveiv:
            self.iv.t.append(self.time[-1])
            self.iv.I.append(self.I)
            self.iv.V.append(self.V_contact)
            self.iv.R.append(self.R)

    def pulse(self, V_arr, duration=None, rate=None, Ilimit=None, maxiter=None):
        ''' Apply voltage pulse to cell.  Ilimit not implemented'''
        if rate is None and duration is None:
            raise Exception('Must give duration XOR rate')
        if duration is not None:
            # Equally space V_arr in time
            t_arr = np.linspace(0, duration, len(V_arr))
        else:
            rate = float(rate)
            t_arr = np.append(0, np.cumsum(np.abs(np.diff(V_arr)/rate)))
            duration = t_arr[-1]

        def V(t):
            return np.interp(t, t_arr, V_arr)

        t0 = self.time[-1]
        i = 0
        while self.time[-1] < t0 + duration:
            if maxiter is not None and i == maxiter:
                break
            # Appends to self.iv
            V_apply = V(self.time[-1] - t0)
            self.compute(V_apply, saveiv=True)

            # Figure out how long next move takes
            pix, dt = self.choose_pixel()
            # enforce some maximum time step, so that voltage has a chance to
            # change even if nothing is happening
            maxtime = duration / 300.
            if dt > maxtime:
                self.time.append(self.time[-1] + maxtime)
                # log that nothing changed in this time
                self.log.append(np.nan)
            else:
                self.toggle(pix)
                self.time.append(self.time[-1] + dt)
            # Print a status every 10 steps
            if i % 10 == 0:
                print('Time: {}'.format(self.iv.t[-1]))
                print('Applied Voltage: {}'.format(V_apply))
                print('Device Voltage: {}'.format(self.iv.V[-1]))
                print('Current: {}'.format(self.iv.I[-1]))
                print('Resistance: {}'.format(self.iv.R[-1]))
                print('Min I to M energy barrier: {}'.format(np.min(self.E_i_m)))
            i += 1

    def plot_neighbors(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        image = np.where(self.neighbormask(), True, np.nan)
        ax.imshow(image, interpolation='nearest', **kwargs)
        plt.draw()

    def plot(self, hue=None, ax=None, hl=None, **kwargs):
        ''' Plot current state of the field '''
        image = np.where(self.mat, 1, np.nan)
        if hue is not None:
            image = image * hue

        if ax is None:
            ax = plt.gca()
        ax.imshow(image, interpolation='nearest', **kwargs)
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

    def plot_grid(self, ax=None, **kwargs):
        ''' plot grid to distinguish pixels or whatever'''
        if ax is None:
            ax = plt.gca()
        ax.hlines(np.arange(self.h) + .5, -.5, self.w - .5, linestyles='dashed',
                  alpha=0.5, **kwargs)
        ax.vlines(np.arange(self.w) + .5, -.5, self.h - .5, linestyles='dashed',
                  alpha=0.5, **kwargs)

    def plotE_vect(self, ax=None, **kwargs):
        ''' plot vector field of E'''
        if ax is None:
            ax = plt.gca()
        X, Y = np.meshgrid(range(self.h), range(self.w), indexing='ij')
        ax.streamplot(Y, X, self.Ey, self.Ex, **kwargs)

    def plotE_quiver(self, ax=None, **kwargs):
        ''' plot vector field of E, one arrow per pixel '''
        if ax is None:
            ax = plt.gca()
        X, Y = np.meshgrid(range(self.h), range(self.w), indexing='ij')
        # Have to invert Ex since quiver is messed up
        ax.quiver(Y, X, self.Ey, -self.Ex, pivot='mid', **kwargs)

    def plotE(self, ax=None, cmap='hot', interpolation=None, **kwargs):
        ''' plot the electric field magnitude'''
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.Emag, cmap=cmap, interpolation=interpolation, **kwargs)

    def resist(self):
        return np.where(self.mat, self.R2, self.R1)

    def Eb(self):
        return np.where(self.mat, self.Eb1, self.Eb0)

    def write(self, fp='mott.pickle'):
        if os.path.isfile(fp):
            print('Pickle file already exists.')
        else:
            with open(fp, 'w') as f:
                pickle.dump(self, f)
                print('Dumped pickle to\n' + fp)


class iv(object):
    ''' Just a little container class for IV data that can do basic plotting'''
    def __init__(self):
        self.t = []
        self.I = []
        self.V = []
        self.R = []

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.V, self.I, '.-', c='SteelBlue')


def solve_heat(Q, nt=1000):
    '''A little poisson solver (static heat eq).
    Not verified to be correct
    What are boundary conditions?
    How to put in thermal conductivity?
    '''
    dx = 1.
    dy = 1.
    p = np.zeros(np.shape(Q))
    pn = np.zeros(np.shape(Q))
    b = -np.abs(Q)
    for n in range(nt):
        pn = p.copy()
        p[1:-1,1:-1] = (dx**2*(pn[1:-1,0:-2] + pn[1:-1,2:]) + \
        dy**2*(pn[0:-2,1:-1] + pn[2:,1:-1]) -
        b[1:-1,1:-1]*dx**2*dy**2)/(2*dx**2+2*dy**2)

        p[0,:] = p[-1,:] = p[:,0] = p[:,-1] = 0.0

    return p



# TODO:  Find a way to generate matrices more quickly by storing previous one
# and making modifications
# also make another version which wraps in the horizontal direction

def solve(R, V_contact=1):
    '''
    Compute currents and voltage of mesh of resistors. Top electrode at 0V
    '''
    # t0 = time()
    out = dict()

    m, n = np.shape(R)

    # Put some contacts on top and bottom
    contact_R = 0.
    contact = contact_R * np.ones((1, n))
    R = np.vstack((contact, R, contact))
    m = m + 2

    Adim = (n - 1) * (m - 1) + 1

    row = []
    col = []
    element = []

    def add_element(r, c, elem):
        row.append(r)
        col.append(c)
        element.append(elem)

    for i in range(m-1):
        for j in range(n-1):
            # Loop numbers
            k = i * (n - 1) + j
            k_above = k - (n - 1)
            k_below = k + (n - 1)
            k_left = k - 1
            k_right = k + 1

            R_above = (R[i, j] + R[i, j+1]) / 2
            R_below = (R[i+1, j] + R[i+1, j+1]) / 2
            R_left = (R[i, j] + R[i+1, j]) / 2
            R_right = (R[i, j+1] + R[i+1, j+1]) / 2

            top = i == 0
            bottom = i == m-2
            left = j == 0
            right = j == n-2

            if top and left:
                add_element(k, k, R_above + R_right + R_left + R_below)
                add_element(k, k_below, -R_below)
                add_element(k, k_right, -R_right)
                add_element(k, Adim-1, -R_left)
            elif top and right:
                add_element(k, k, R_above + R_right + R_left + R_below)
                add_element(k, k_below, -R_below)
                add_element(k, k_left, -R_left)
            elif bottom and left:
                add_element(k, k, R_above + R_right + R_left + R_below)
                add_element(k, k_above, -R_above)
                add_element(k, k_right, -R_right)
                add_element(k, Adim-1, -R_left)
            elif bottom and right:
                add_element(k, k, R_above + R_right + R_left + R_below)
                add_element(k, k_above, -R_above)
                add_element(k, k_left, -R_left)
            elif top:
                add_element(k, k, R_above + R_right + R_left + R_below)
                add_element(k, k_below, -R_below)
                add_element(k, k_left, -R_left)
                add_element(k, k_right, -R_right)
            elif bottom:
                add_element(k, k, R_above + R_right + R_left + R_below)
                add_element(k, k_above, -R_above)
                add_element(k, k_left, -R_left)
                add_element(k, k_right, -R_right)
            elif left:
                add_element(k, k, R_above + R_right + R_left + R_below)
                add_element(k, k_above, -R_above)
                add_element(k, k_below, -R_below)
                add_element(k, k_right, -R_right)
                add_element(k, Adim-1, -R_left)
            elif right:
                add_element(k, k, R_above + R_right + R_left + R_below)
                add_element(k, k_above, -R_above)
                add_element(k, k_below, -R_below)
                add_element(k, k_left, -R_left)
            else:
                add_element(k, k, R_above + R_right + R_left + R_below)
                add_element(k, k_above, -R_above)
                add_element(k, k_below, -R_below)
                add_element(k, k_left, -R_left)
                add_element(k, k_right, -R_right)

    # Put V_contact across top and bottom
    for i in range(m-1):
        R_edge = (R[i, 0] + R[i+1, 0]) / 2
        k = i * (n - 1)
        add_element(Adim-1, k, -R_edge)
        add_element(Adim-1, Adim-1, R_edge)
    b = np.zeros(Adim)
    b[-1] = V_contact

    #t1 = time()
    #print('Time to generate matrix {}'.format(t1 - t0))

    A = coo_matrix((element, (row, col)), shape=(Adim, Adim))
    A = A.tocsr()

    #t2 = time()
    #print('Time to generate sparse matrix {}'.format(t2 - t1))

    x = spsolve(A, b)

    #t3 = time()
    #print('Time to solve sparse matrix equation {}'.format(t3 - t2))

    # Calculate node currents from loop currents
    I_loop = x[:-1].reshape((m-1, n-1))
    I_vert = np.hstack((x[-1]*np.ones((m-1, 1)), I_loop)) - np.hstack((I_loop, np.zeros((m-1, 1))))
    #zero_hor = np.zeros(1, n-1)
    #I_hor = np.vstack(zero_hor, I_loop) - np.vstack(I_loop, zero_hor)
    I_hor = np.vstack((I_loop[0], np.diff(I_loop, axis=0), I_loop[-1]))
    R_vert = R[:-1, :] / 2. + R[1:, :] / 2.
    #I_node_vert = np.vstack((I_vert[0], np.diff(I_vert, axis=0), I_vert[-1]))
    #I_node_hor = np.hstack((I_hor[:, [0]], np.diff(I_hor, axis=1), I_hor[:, [-1]]))
    #I_vect = np.dstack((I_node_vert, I_node_hor))

    I_node_vert = np.vstack((I_vert[0], I_vert[:-1]/2 + I_vert[1:]/2, I_vert[-1]))
    I_node_hor = np.hstack((I_hor[:,[0]], I_hor[:,:-1]/2 + I_hor[:,1:]/2, I_hor[:,[-1]]))
    I_squared_vert = np.vstack((I_vert[0]**2, I_vert[:-1]**2 + I_vert[1:]**2, I_vert[-1]**2))
    I_squared_hor = np.hstack((I_hor[:,[0]]**2, I_hor[:,:-1]**2 + I_hor[:,1:]**2, I_hor[:,[-1]]**2))
    I_squared = I_squared_vert + I_squared_hor
    I_mag = np.sqrt(I_squared)
    P = I_squared * R/2.

    # Total current at 1V
    I_tot = np.sum(I_vert[0])
    R_tot = V_contact/I_tot

    # Calculate voltages from vertical currents
    V = V_contact + np.vstack((np.zeros((1, n)), np.cumsum(-I_vert * R_vert, axis=0)))

    # Electric field in volts/pixel!
    E = np.gradient(-V)
    Ex = E[0]
    Ey = E[1]
    E_mag = np.sqrt(Ex**2 + Ey**2)

    out['V'] = V[1:-1,:]
    out['R'] = R_tot
    out['I'] = I_tot
    out['Ix'] = I_node_vert[1:-1,:]
    out['Iy'] = I_node_hor[1:-1,:]
    out['I_mag'] = I_mag[1:-1,:]
    out['P'] = P[1:-1,:]
    out['E'] = E_mag[1:-1,:]
    out['Ex'] = Ex[1:-1,:]
    out['Ey'] = Ey[1:-1,:]

    #t4 = time()
    #print('Time to get all parameters out of solution {}'.format(t4 - t3))

    return out

class rerun(mott):
    ''' Use history of another instance to rerun simulation '''
    def __init__(self, parent, start=0, end=-1):
        mott.__init__(self, h=parent.h, w=parent.w, R1=parent.R1, R2=parent.R2)
        self.commands = parent.log[:end]
        self.frame = 0
        self.next(start)

    def next(self, numframes=1):
        # Execute numframes of history
        for p in self.commands[self.frame:self.frame + numframes]:
            if type(p) == tuple:
                self.toggle(p)
        self.frame += numframes
        #print('Frame # ' + str(self.frame))
        return 0 if self.frame > len(self.commands) else 1

def movie(game, interval=0.1, skipframes=0, start=0):
    ''' TODO: generate everything before animation somehow '''
    # PROBLEM!!!  Does not clear previous frame!
    t = rerun(game, start=start)
    def data_gen():
        while t.frame < len(t.commands):
            yield t
            t.next(1 + skipframes)

    def run(data):
        try:
            del ax.images[0]
        except:
            pass
        data.plot(ax=ax)
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

def write_frames(mott, dir, skipframes=0, start=0, end=-1, dV=None, dt=None, plotxy=None, plotE=True, cmap=None, vmax=.00222, **kwargs):
    '''
    Write a bunch of pngs to directory for movie.

    Can plot electric field in addition to occupation array, should extend to plot any quantity

    Can make a simultaneous xy plot below:
    plotxy = ('xdata', 'ydata') should be an attribute name of the iv container class

    start, end may be given in simulation steps

    set dV to step frames equally in voltage
    set dt to step frames equally in time
    '''
    plt.ioff()
    if cmap is None:
        cmap = truncate_colormap(plt.cm.inferno, .2, 1.0)

    #import pdb; pdb.set_trace()
    if dV is not None:
        # Determine frames to save
        # Convert all changes to positive for purposes of interpolation
        vsteps = np.cumsum(np.abs(np.diff(mott.iv.V[start:end])))
        # Groups all of the voltages according the which dV increment they are in
        # gp = groupby(enumerate(vsteps), lambda vs: int(vs[1]/dV))
        # ind = [0]
        # ind.extend([g.next()[0] + 1 for k,g in gp])
        # this has problem if there is no data point in every dV range
        # Should repeat indices when there is no data point in a dV range

        # This is an array which says how many times the frame corresponding to
        # the ith element of iv.V should appear
        num_reps = diff(np.int8(vsteps/dV))
        ind = [0]
        ind.extend(flatten([[i]*j for i,j in enumerate(num_reps)]))
    elif dt is not None:
        # time should not have negative diffs, but do this anyway I guess
        tsteps = np.cumsum(np.abs(np.diff(mott.iv.t[start:end])))
        num_reps = diff(np.int8(tsteps/dt))
        ind = [0]
        ind.extend(flatten([[i]*j for i,j in enumerate(num_reps)]))
    else:
        ind = np.arange(start, len(mott.log), skipframes + 1)

    if not os.path.isdir(dir):
        os.makedirs(dir)

    r = rerun(mott, start=start, end=end)
    def data_gen():
        for di in np.diff(ind):
            #print('Take {} steps'.format(di))
            yield r
            # Don't run next() if di is zero!
            if di != 0 and not r.next(di):
                return

    if plotxy is not None:
        # TODO: Size should depend on aspect ratio of cell
        fig = plt.figure(figsize=(9, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        xattr, yattr = plotxy
        ax2.set_xlabel(xattr)
        ax2.set_ylabel(yattr)
        # Plot all data for autorange
        xdata = getattr(mott.iv, xattr)
        ydata = getattr(mott.iv, yattr)
        ax2.plot(xdata[start:end], ydata[start:end])
        ax2.scatter(xdata[start], ydata[start])
        # del ax2.lines[0]
    else:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)

    # Don't know why -1
    length = len(ind) - 1
    for i, d in enumerate(data_gen(), 1):
        d.compute(mott.iv.V[i])
        ax1.cla()
        if plotE:
            d.plotE(cmap='plasma', ax=ax1, vmin=0, vmax=10)
            # Temp
            #ax1.imshow(T, cmap='hot')

        d.plot(hue=d.I_mag, cmap=cmap, ax=ax1, vmin=0, vmax=vmax, **kwargs)
        if plotxy:
            del ax2.lines[0]
            del ax2.collections[0]
            xpartial = xdata[start:ind[i]]
            ypartial = ydata[start:ind[i]]
            if len(xpartial) == 0:
                # At least plot first point
                xpartial = xdata[start:start+1]
                ypartial = ydata[start:start+1]
            ax2.plot(xpartial, ypartial, c='SteelBlue', alpha=.8)
            ax2.scatter(xpartial[-1], ypartial[-1], c='black', zorder=2)
        fn = os.path.join(dir, 'frame{:0>4d}.png'.format(i))
        fig.savefig(fn, bbox_inches='tight')
        if i == length:
            # Write image again as the 0000th frame so that it shows up as
            # preview.
            fig.savefig(os.path.join(dir, 'frame0000.png'), bbox_inches='tight')
        print('Wrote {}/{}: {}'.format(i, length, fn))

    plt.close(fig)

    frames_to_mp4(dir)
    mott.write(os.path.join(dir, 'data.pickle'))

    plt.ion()

def frames_to_mp4(directory):
    # Send command to create video with ffmpeg
    cmd = (r'cd "{}" & ffmpeg -framerate 30 -i frame%04d.png -c:v libx264 '
            '-r 30 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            'out.mp4').format(directory)
    os.system(cmd)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
