alpha = 100
beta = 2
Eb = linspace(.3, .9, 6)

data = []

for E in Eb:
    f = fractal(50, 100)
    f.alpha = alpha
    f.beta = beta
    f.Eb = E
    data.append(f)
    f.pulse([0, 100], rate=1e8, Ilimit=.0003, maxiter=2000)
    vmax = f.V_contact / (f.h * f.R2) * .8
    directory = 'Video\\Ebtest\\Ebtest_' + str(E)
    write_frames(f, directory, plotE=False, skipframes=10, vmax=vmax)
    f.write(os.path.join(directory, 'data.pickle'))
