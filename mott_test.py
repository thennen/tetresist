import fnmatch
import shutil

# Try to run a series of mott simulations like France paper
ma = []
vpulse = linspace(350, 600, 10)

# meaningless duration due to choice of parameters
duration = 1e6

# construct pulse array with long wait at zero
ttotal = 2000
tstart = 100
tpulse = 400
pulse = np.zeros(ttotal)
pulse[tstart:tstart + tpulse] = 1.

maxiter = 15000

for v in vpulse:
    m = mott()
    m.pulse(v * pulse, duration=duration, maxiter=maxiter)
    ma.append(m)

# Make a plot
fig1, ax1 = plt.subplots()
cm = plt.cm.rainbow
for i, (m,v) in enumerate(zip(ma, vpulse)):
    plot(m.iv.t, m.iv.R, label='{:.1f}'.format(v), c=cm(float(i)/len(ma)))
legend(title='Pulse Voltage (AU)')
ax1.set_xlabel('\"Time\"')
ax1.set_ylabel('\"Resistance\" = V/I')

# If you made it this far, save data

# TODO: save data during simulation, then it can be inspected by another
# kernel, and there will be data even if simulation stops unexpectedly 

# Make a new directory with same name as script, with number appended
data_folder = 'sim_data'
scriptfile = __file__
scriptname = os.path.splitext(scriptfile)[0]
existingdirs = fnmatch.filter(os.listdir('sim_data'), 'scriptname_???')
if any(existingdirs):
    numbers = [int(ed.split('_')[-1]) for ed in existingdirs]
    nextn = max(numbers) + 1
else:
    nextn = 0
foldername = '{}_{:03d}'.format(scriptname, nextn)
folderpath = os.path.join(data_folder, foldername)
os.makedirs(folderpath)
print('Saving simulation data to {}'.format(folderpath))

# Copy script into the folder
shutil.copy(scriptfile, os.path.join(folderpath, scriptfile))
# Copy this dependency as well, in case of poor git practice
shutil.copy('mott.py', os.path.join(folderpath, 'mott.py'))

# Save all sim instances
# Varied parameter in filename
param_name = 'Voltage_Pulse'
param_values = vpulse
for (m, p) in zip(ma, param_values):
    picklefilename = '{}_{}.pickle'.format(param_name, p)
    picklepath = os.path.join(folderpath, picklefilename)
    m.write(picklepath)

fig1_fname = 'R_vs_time.png'
fig1.savefig(os.path.join(folderpath, fig1_fname))

# Write the time each simulation took
# Maybe write fixed parameters somewhere

