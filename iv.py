import numpy as np
from matplotlib import pyplot as plt
from tetresist import tetresist

# TODO:
#       current limiting
#       better plots for preview
#       wrap horizontal
#       filter IV loop to simulate measurement
#       Detached volumes of metal should not reduce impacting pieces

kT = 0.026
f0 = 10**13
# not used yet
#desorb_Eb = .3
#diffuse_Eb = .15
#_Eb = .35
#alpha = .5
#beta = .5

reduced_Eb = .45
move_Eb = .15
beta = .3

I_limit = .1

t = tetresist(50, 100, R1=10000, R2=1)

V_wfm = [0]
t_wfm = [0]

# These get new values whenever compute() happens, which is not every iteration
V = [0]
I = [0]
R = []
IVtime = [0]
time = [0]
step = [0]
current_time = 0

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
                did_something = True
                t.tetras[-1].Eb = move_Eb
                print('Spawn new tetra')
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
                    if t.tetras[m[0]].Eb == reduced_Eb:
                        # hit reduced tetra
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
    spawn_Eb = .35 - .1 * E_electrode

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
    gamma_spawn = f0 * np.exp(-spawn_Eb / kT)
    gamma = np.concatenate((gamma_left, gamma_right, gamma_up, gamma_down, gamma_spawn))

    return gamma

def preview():
    ''' do a plot showing state of system '''
    ax1.cla()
    t.plot(hue=t.I_mag, cmap='Reds', ax=ax1)
    t.plotV(alpha=.4, cmap='Blues', ax=ax1)
    #t.plotE_vect(ax=ax1)
    ax1.set_title('V = {:.4e}, t = {:.4e}'.format(V_interp(current_time), current_time))
    ax2.cla()
    #ax2.plot(t.Ex[0])
    #ax2.plot(t.Ex[1])
    ax2.plot(V, I)
    ax2.set_title('IV Loop')
    ax2.set_xlabel('V')
    ax2.set_ylabel('I')

    #E_electrode = t.Ex[0]
    #spawn_Eb = .3 - .1 * E_electrode
    #ax3.cla()
    #ax3.plot(f0 * np.exp(-spawn_Eb / kT))

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
            V.append(V_contact)
            I.append(t.I)
            R.append(t.R)
            IVtime.append(time[-1])
            # Stop if current limit exceeded
            if t.I >= I_limit:
                # Quick hack to stop pulse
                current_time = duration + t0
        if i % 500 == 0:
            preview()
        print('Time: {:.3e} s'.format(time[-1]))
        print('V_contact: {:.3e} V'.format(V_contact))
        print('R: {:.3e} Ohms'.format(R[-1]))
        print('Tetras: {}'.format(len(t.tetras)))
        i += 1
    t.compute()
    preview()

if __name__ == '__main__':
    t.compute(V_contact=1)
    R.append(t.R)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,12))

    #fig1, ax1 = plt.subplots()
    #fig2, ax2 = plt.subplots()
    #fig3, ax3 = plt.subplots()
    preview()
