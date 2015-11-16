import numpy as np
import subprocess
from scipy import linalg
from matplotlib import pyplot as plt
import time
import sim
plt.ion()

# Try to solve resistance between two points on a mesh


#t = sim.tetresist(20, 10)
#t.randgame()
figure()
t.plot()

t0 = time.time()
m, n = np.shape(t.field)

pt1 = (0, 0)
pt2 = (m+1, n-1)

# Array of resistances
#R = np.ones(( m, n ))
R = sim.resist(t.field, 10000, 10)

# Put some contacts on top and bottom
contact_R = 0.0001
contact = contact_R * np.ones((1, n))
R = np.vstack(( contact, R, contact ))
m = m + 2

# Calculate resistances between points
# Horizontal resistances
R_h = R[:, :-1] / 2. + R[:, 1:] / 2.
# Vertical resistances
R_v = R[:-1, :] / 2. + R[1:, :] / 2.

t1 = time.time()

print('Seconds to calculate resistances between pts: {}'.format(t1- t0))

# Map nodes and edges to integers
def node_num(i, j):
    # Left to right, top to bottom
    return n * i + j
def vedge_num(i, j):
    # Left to right, top to bottom
    return n * i + j
def hedge_num(i, j):
    # Left to right, top to bottom
    return n * (m-1) + (n-1) * i + j
# TODO: probably will need inverse functions

t2 = time.time()

# Generate netlist
def connect_nodes(resistor, n1, n2, value):
    # Output with format R# n1 n2 value
    # Might need to zfill, try without
    #resistor = str(resistor).zfill(4)
    #n1 = str(n1).zfill(4)
    #n2 = str(n2).zfill(4)
    return 'R{} N{} N{} {}\n'.format(resistor+1, n1+1, n2+1, value)
netlist = []
for i in range(shape(R_v)[0]):
    for j in range(shape(R_v)[1]):
        rnum = vedge_num(i, j)
        n1 = node_num(i, j)
        n2 = node_num(i+1, j)
        R = R_v[i, j]
        netlist.append(connect_nodes(rnum, n1, n2, R))
for i in range(shape(R_h)[0]):
    for j in range(shape(R_h)[1]):
        rnum = hedge_num(i, j)
        n1 = node_num(i, j)
        n2 = node_num(i, j+1)
        R = R_h[i, j]
        netlist.append(connect_nodes(rnum, n1, n2, R))
# Ground something - could have done it beforehand
gnd = m*n
netlist = [c.replace('N{}'.format(gnd), '0') for c in netlist]
# Add current source
netlist.append('I1 N1 0 1\n')
#netlist.append('V1 N1 0 1\n')
# Find dc operating point
netlist.append('.op\n')
netlist.append('.savebias spiceout.txt')
t3 = time.time()

print('Seconds to generate netlist: {}'.format(t3 - t2))

# Write to disk
netlist_fp = r'C:\Users\thenn\Desktop\tetris\netlist.sp'
with open(netlist_fp, 'w') as f:
    f.writelines(netlist)

# Send to spice
t4 = time.time()
spice_path = 'C:\Program Files (x86)\LTC\LTspiceIV\scad3.exe'
subprocess.call('{} -b -run {}'.format(spice_path, netlist_fp))
t5 = time.time()
print('Seconds for spice to compute result: {}'.format(t5 - t4))


# parse into matrix
with open(r'C:\Users\thenn\Desktop\tetris\spiceout.txt') as f:
    spiceout = f.read()
node_v = [s.replace('+','').strip().split(')=') for s in spiceout.split('V(n')]
node_v = node_v[1:]
V = np.empty(len(node_v))
for node, v in node_v:
    try:
        V[int(node) - 1] = float(v)
    except:
        # for some reason the last one has issues
        V[int(node) - 2] = float(v)
# gnd pt not included, just append it for now
V = np.append(V, 0)
V = reshape(V, (m,n))

# Plot it
plt.figure()
print m, n
#plt.imshow(-V, interpolation='none')


# Electric field
E = np.gradient(V)
Ey = E[0]
Ex = E[1]
Emag = [sqrt(x**2 + y**2) for x,y in zip(*E)]

plt.figure()
plt.imshow(Emag, interpolation='none')

# Currents
# y = -1 * CA * matrix(x).T


