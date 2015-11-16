import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sim

def data_gen():
    p = 0.1
    t = sim.tetresist(100, 40)
    t.spawn()
    while 1:
        while not t.gameover:
            while t.randmove(p=p, x=1) == 1:
                yield t
            print('spawning')
            t.spawn()
        print('hit top')
        for _ in range(80):
            yield t
        p = 0.2
        n = 12
        # make n fly away
        top_pos = [tetra.loc[0] + tetra.top for tetra in t.tetras]
        for _ in range(n):
            m = t.highest()
            print('tetra ' + str(m) + ' flying away')
            while t.tetras[m].loc[0] > 0:
                t.randmove(tetranum=m, p=.1, x=-1)
                yield t
            t.tetras.pop(m)
            yield t
        t.gameover = False
        for _ in range(50):
            yield t


def run(data):
    try:
        del ax.images[0]
    except:
        pass
    data.plot(ax)
    return ax.images

Writer = animation.writers['ffmpeg']
writer = Writer(fps=150, metadata=dict(artist='Me'), bitrate=3000)
fig, ax = plt.subplots()

ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=1, save_count=5000)
ani.save('im.mp4', writer=writer)
