alpha = [0, 10, 100, 1000, 10000, 100000]
beta = 2
Eb = .3

data = []

for a in alpha:
    f = fractal(50, 100)
    f.alpha = a
    f.beta = beta
    f.Eb = Eb
    data.append(f)
    f.pulse([0, 2], rate=1e8, Ilimit=.0003, maxiter=2000)
    directory = 'Video\\alphatest\\alphatest_' + str(a)
    write_frames(f, directory, plotE=False, skipframes=10, vmax=.00001)
    f.write(os.path.join(directory, 'data.pickle'))
