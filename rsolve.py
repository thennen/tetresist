import numpy as np
from scipy import linalg
from scipy import sparse
from matplotlib import pyplot as plt
import time
import sim
plt.ion()

# Try to solve resistance between two points on a mesh


#t = sim.tetresist(20, 10)
#t.randgame()
#figure()
#t.plot()

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

A = np.zeros((2 * m * n - m - n, m * n), dtype=int)
# Connect everything but last row and column
for i in range(m):
    for j in range(n):
        if i < m-1:
            A[vedge_num(i, j), node_num(i, j)] = 1#/R_v[i, j]
        if i > 0:
            A[vedge_num(i-1, j), node_num(i, j)] =  - 1#/R_v[i-1, j]
        if j < n-1:
            A[hedge_num(i, j), node_num(i, j)] = 1#/R_h[i, j]
        if j > 0:
            A[hedge_num(i, j-1), node_num(i, j)] = -1#/R_h[i, j-1]

A = np.matrix(A)

t3 = time.time()

print('Seconds to generate connectivity matrix: {}'.format(t3 - t2))
# Create conductivity matrix
#C = np.identity(2 * m * n - m - n) / r
C = np.diag(np.concatenate(((1./R_v).flatten(), (1./R_h).flatten())))

C = np.matrix(C)

CA = C * A
AtCA = A.transpose() * CA

# Ground pt1 by deleting its column/row
#AtCA_gnd = np.delete(np.delete(AtCA, pt1[0], 0), pt1[1], 1)
AtCA_gnd = np.delete(np.delete(AtCA, node_num(*pt1), 1), node_num(*pt1), 0)

(lu, piv) = linalg.lu_factor(AtCA_gnd)

b = np.zeros(m * n - 1)
# 1 amp through the pt2
b[node_num(*pt2) - 1] = 1

x = linalg.lu_solve((lu, piv), b)
t4 = time.time()
print('R = {} Ohm.  AtCA Computation took {} seconds'.format(x[-1], t4 - t0))

# Put gnd back in and imshow voltage
x = np.concatenate((x[:node_num(*pt1)], [0], x[node_num(*pt1):]))
V = reshape(x, (m,n))
fig, ax = plt.subplots()
ax.imshow(V, interpolation='none')

# Electric field
E = np.gradient(V)
Ey = E[0]
Ex = E[1]
Emag = [sqrt(x**2 + y**2) for x,y in zip(*E)]
fig, ax = plt.subplots()
ax.imshow(Emag, interpolation='none')

# Currents
# y = -1 * CA * matrix(x).T


# Try sparse algorithms

A = np.zeros((2 * m * n - m - n, m * n), dtype=int)
# Connect everything but last row and column
for i in range(m):
    for j in range(n):
        if i < m-1:
            A[vedge_num(i, j), node_num(i, j)] = 1#/R_v[i, j]
        if i > 0:
            A[vedge_num(i-1, j), node_num(i, j)] =  - 1#/R_v[i-1, j]
        if j < n-1:
            A[hedge_num(i, j), node_num(i, j)] = 1#/R_h[i, j]
        if j > 0:
            A[hedge_num(i, j-1), node_num(i, j)] = -1#/R_h[i, j-1]

A = np.matrix(A)
