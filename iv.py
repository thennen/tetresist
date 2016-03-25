import numpy as np
from matplotlib import pyplot as plt
import tetresist
from itertools import groupby
import os

# TODO:
#       wrap horizontal
#       filter IV loop to simulate measurement
#       Detect switching and change sweep direction
#       input rate instead of time
#       Use real units
#       Stop program halt due to trapped pieces
#       Replace "spawn" and "reduced_Eb" to equivalent oxidation_Eb


# not used yet
#desorb_Eb = .3
#diffuse_Eb = .15
#_Eb = .35
#alpha = .5
#beta = .5

kT = 0.026
f0 = 10**13
# Spawn energy barrier = spawn_Eb - alpha * E_electrode
spawn_Eb = .35
alpha=.1
# Moving energy barrier for ions move_Eb - beta * E
move_Eb = .18
beta = .3
# Moving energy barrier for "reduced" ion: reduced_Eb - beta * E
reduced_Eb = .45


I_limit = .1

t = tetresist.tetresist(50, 100, R1=10000, R2=1)

V_wfm = [0]
t_wfm = [0]
time = [0]
current_time = 0

# These get new values whenever compute() happens, which is not every iteration
V = [0]
I = [0]
R = []
frame = [0]
IVtime = [0]

#wtf_show_p = 1


def V_interp(t):
    ''' calculate voltage at some time '''
    return np.interp(t, t_wfm, V_wfm)
# return np.interp(t, t_wfm, V_wfm, left=0, right=0)


def tetrafield(tetra):
    ''' return average field in x and y direction for tetra '''
    # Create field above to push pieces down, otherwise wrap happens
    # E_above = (0.1, 0)
    E_above = (V_interp(current_time) / t.h, 0)
    o = list(tetra.occupied)
    leno = len(o)
    pos = [p for p in o if p[0] >= 0]
    lenp = len(pos)
    Ex_avg = (np.sum(t.Ex[zip(*pos)]) + (leno - lenp)*E_above[0]) / leno
    Ey_avg = (np.sum(t.Ey[zip(*pos)]) + (leno - lenp)*E_above[1]) / leno
    return Ex_avg, Ey_avg

def tetravoltage(tetra):
    ''' return average voltage for tetra '''
    o = list(tetra.occupied)
    leno = len(o)
    pos = [p for p in o if p[0] >= 0]
    lenp = len(pos)
    V_above = np.mean(t.V[0])
    V_avg = (np.sum(t.V[zip(*pos)]) + (leno - lenp) * V_above) / leno
    return V_avg

def tetrapower(tetra):
    ''' return average power at tetra location'''
    o = list(tetra.occupied)
    P_avg = np.sum(t.P[zip(*o)])/len(o)
    return P_avg

def do_something(t):
    ''' calculate all rate parameters and decide to do something.  return dt '''
    # Probably very slow
    #global wtf_show_p

    # Flag to decide whether to recompute voltages
    compute = False

    ntetras = len(t.tetras)

    # Find probabilities for possible events
    gamma = calc_gamma(t)
    p, dt = calc_p_dt(gamma)

    #if wtf_show_p % 10 == 1:
        #ax3.cla()
        #ax3.plot(p)
        #plt.pause(.1)
    #wtf_show_p += 1


    # Pick the something that will happen
    did_something = False
    while not did_something:
        something = np.random.choice(range(len(gamma)), p=p)
        if (ntetras == 0) or something >= 4 * ntetras:
            # spawn new tetra
            spawn_loc = something - 4 * ntetras
            if t.spawn(loc=(-4, spawn_loc)):
            #if t.spawn(loc=(-4, spawn_loc), kind='single'):
                did_something = True
                t.tetras[-1].Eb = move_Eb
                # print('Spawn new tetra')
            else:
                # Failed to spawn
                gamma[something] = 0
                p, dt = calc_p_dt(gamma)
        else:
            direction = something / ntetras
            tetranum = something % ntetras
            # If reduced piece moved, consider it oxidized
            if t.tetras[tetranum].Eb == reduced_Eb:
                t.tetras[tetranum].Eb = move_Eb
                compute = True

            # Try to move existing tetra
            if direction == 3:
                m = t.move(1, 0, tetranum=tetranum)
            elif direction == 2:
                m = t.move(-1, 0, tetranum=tetranum)
            elif direction == 1:
                m = t.move(0, 1, tetranum=tetranum)
            elif direction == 0:
                m = t.move(0, -1, tetranum=tetranum)
            # May have deleted the tetra if it hit something and wasn't visible
            # This is a legitimate move
            if len(t.tetras) < ntetras:
                m = 0

            if m:
                # Hit something
                # TODO: Should count as an action if it hasn't happened before
                # Prevent that from happening again, and recalculate dt
                gamma[something] = 0
                p, dt = calc_p_dt(gamma)
                if direction == 3 and m[0] == -1:
                    # hit bottom
                    t.tetras[tetranum].Eb = reduced_Eb

                if m[0] != -1:
                    # hit tetra
                    #if t.tetras[m[0]].Eb == reduced_Eb:
                        # hit reduced tetra
                        #t.tetras[tetranum].Eb = reduced_Eb

                    V_contact = V_interp(current_time)
                    vplus = np.max(V_contact, 0)
                    vminus = np.min(V_contact, 0)
                    vthresh = vminus + 0.1 * (vplus - vminus)
                    threshold = tetravoltage(t.tetras[m[0]]) < vthresh
                    isreduced = t.tetras[m[0]].Eb == reduced_Eb
                    if isreduced and threshold:
                        # Voltage is 'near' more negative electrode
                        # impacted piece is reduced
                        # TODO: more sophisticated decision about when to reduce
                        t.tetras[tetranum].Eb = reduced_Eb

                compute = True
            else:
                # Moved successfully
                did_something = True
                if direction == 3:
                    print('Move down')
                elif direction == 2:
                    print('Move up')
                elif direction == 1:
                    print('Move right')
                elif direction == 0:
                    print('Move left')

    return (dt, compute)


def calc_p_dt(gamma):
    gamma_sum = np.sum(gamma)
    dt = - 1 / gamma_sum * np.log(np.random.rand())
    p = gamma / gamma_sum
    return p, dt


def calc_gamma(t):
    ''' Calculate the probability for every possible movement '''
    # Probably very slow

    E_electrode = t.Ex[0]

    Ex = []
    Ey = []
    Eb = []
    for tet in t.tetras:
        E = tetrafield(tet)
        Ex.append(E[0])
        Ey.append(E[1])
        Eb.append(tet.Eb)

    Ex = np.array(Ex)
    Ey = np.array(Ey)

    # Make a big 1D array of probabilities of things that can happen
    gamma_left = f0 * np.exp(-(Eb + beta * Ey) / kT)
    gamma_right = f0 * np.exp(-(Eb - beta * Ey) / kT)
    gamma_up = f0 * np.exp(-(Eb + beta * Ex) / kT)
    gamma_down = f0 * np.exp(-(Eb - beta * Ex) / kT)
    gamma_spawn = f0 * np.exp(-(spawn_Eb - alpha * E_electrode) / kT)
    gamma = np.concatenate((gamma_left, gamma_right, gamma_up, gamma_down, gamma_spawn))

    return gamma

def preview():
    ''' do a plot showing state of system '''
    ax1.cla()
    t.plot(hue=t.I_mag, cmap='Reds', ax=ax1)
    #t.plotV(alpha=.4, cmap='Blues', ax=ax1)
    t.plotE(alpha=.4, cmap='Blues', ax=ax1)
    #t.plotE_vect(ax=ax1)
    ax1.set_title('V = {:.4e}, t = {:.4e}'.format(V_interp(current_time), current_time))
    ax2.cla()
    #ax2.plot(t.Ex[0])
    #ax2.plot(t.Ex[1])
    ax2.plot(V, I)
    ax2.set_title('IV Loop')
    ax2.set_xlabel('V')
    ax2.set_ylabel('I')

    plt.show()
    plt.pause(.4)


def pulse(v=[0, 5, 0, -5, 0], duration=1e-3):
    ''' Apply a voltage pulse to t '''
    # TODO: Maybe add some stop condition
    global current_time
    t0 = current_time
    V_wfm.extend(v)
    t_wfm.extend(np.linspace(t0, t0 + duration, len(v)))
    i = 0
    while current_time - t0 < duration:
        # Save some stuff
        dt, compute = do_something(t)
        current_time += dt
        time.append(current_time)
        V_contact = V_interp(current_time)
        # Limit current
        #V_contact = min(V_contact, I_limit*t.R)
        if compute or i % 10 == -1:
            # compute sometimes
            t.compute(V_contact=V_contact)
            if t.I >= I_limit:
                # if current limit exceeded, scale a bunch of computed
                # parameters
                factor = I_limit / t.I
                t.I = I_limit
                t.I_mag *= factor
                t.V *= factor
                t.P *= factor
                t.Ex *= factor
                t.Ey *= factor
                t.Ix *= factor
                t.Iy *= factor
                # Stop if current limit exceeded
                # Quick hack to stop pulse
                # current_time = duration + t0
            I.append(t.I)
            V.append(V_contact)
            R.append(t.R)
            frame.append(len(t.history))
            IVtime.append(time[-1])
        if i % 500 == 0:
            preview()
        print('Time: {:.3e} s'.format(time[-1]))
        print('V_contact: {:.3e} V'.format(V_contact))
        print('R: {:.3e} Ohms'.format(R[-1]))
        print('Tetras: {}'.format(len(t.tetras)))
        i += 1
    t.compute()
    preview()

def writedata(fp='iv'):
    ''' Write IV loop data '''
    # Why writing in text?
    noext = os.path.splitext(fp)[0]
    txtfp = noext + '.txt'
    picklefp = noext + '.pickle'
    if os.path.isfile(fp):
        print('File already exists.')
    else:
        with open(txtfp, 'w') as f:
            f.write('#kT = {}/n'.format(kT))
            f.write('#f0 = {}/n'.format(f0))
            f.write('#spawn_Eb = {}/n'.format(spawn_Eb))
            f.write('#alpha = {}/n'.format(alpha))
            f.write('#move_Eb = {}/n'.format(move_Eb))
            f.write('#beta = {}/n'.format(beta))
            f.write('#reduced_Eb = {}/n'.format(reduced_Eb))
            f.write('frame\tI\tV\tR\ttime\n')
            np.savetxt(f, np.transpose([frame, I, V, R, IVtime]),
                       fmt='%.3e', delimiter='\t')
            print('Wrote data to\n' + fp)
            t.pickle(picklefp)

def write_frames(dir, skipframes=0, start=0, dV=.01, writeloop=True):
    '''
    Write a bunch of pngs to directory for movie.
    Step frames by voltage
    basically so you don't capture a ton of frames where nothing is happening
    TODO: step by voltage OR current, so you don't miss any good frames
    '''
    plt.ioff()

    # Determine frames to save
    vsteps = np.cumsum(np.abs(np.diff(V)))
    # Doesn't catch frame if vstep is not much smaller than dV
    #big_steps = np.where(diff(vsteps % dV) < 0)[0]
    #ind = big_steps + 2
    gp = groupby(enumerate(vsteps), lambda vs: int(vs[1]/dV))
    ind = [0]
    ind.extend([g.next()[0] + 1 for k,g in gp])
    save_frames = np.array(frame)[ind]

    if not os.path.isdir(dir):
        os.makedirs(dir)

    r = tetresist.rerun(t, start=start)
    def data_gen():
        for di in np.diff(save_frames):
            yield r
            if not r.next(di):
                return

    if writeloop:
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax2.set_xlabel('V')
        ax2.set_ylabel('I')
        # Plot loops for autorange
        ax2.plot(V, I)
        ax2.scatter(0,0)
        #del ax2.lines[0]
    else:
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)

    length = len(ind)
    for i, d in enumerate(data_gen()):
        d.compute()
        ax1.cla()
        d.plotE(alpha=.2, cmap='Oranges', ax=ax1)
        d.plot(hue=d.I_mag, cmap='Reds', ax=ax1)
        if writeloop:
            del ax2.lines[0]
            del ax2.collections[0]
            Vloop = V[:ind[i]]
            Iloop = I[:ind[i]]
            if len(Vloop) == 0:
                Vloop = [0]
                Iloop = [0]
            ax2.plot(Vloop, Iloop, c='SteelBlue', alpha=.8)
            ax2.scatter(Vloop[-1], Iloop[-1], c='black', zorder=2)
        fn = os.path.join(dir, 'frame{:0>4d}.png'.format(i))
        fig.savefig(fn, bbox_inches='tight')
        print('Wrote {}/{}: {}'.format(i, length, fn))

    plt.close(fig)
    # Send command to create video with ffmpeg
    #os.system(r'ffmpeg -framerate 30 -i loop%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4')

    plt.ion()


if __name__ == '__main__':
    t.compute(V_contact=1)
    R.append(t.R)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,12))

    #fig1, ax1 = plt.subplots()
    #fig2, ax2 = plt.subplots()
    #fig3, ax3 = plt.subplots()
    preview()
