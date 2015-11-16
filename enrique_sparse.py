import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
import time
import sim

plt.ion()

#t = sim.tetresist(20, 10)
#t.randgame()
#figure()
#t.plot()

t0 = time.time()
m, n = np.shape(t.field)

# Array of resistances
#R = np.ones(( m, n ))
#R = sim.resist(t.field, 10000, 10)
R = sim.resist(t.field, 10000, 10)

# Put some contacts on top and bottom
contact_R = 0.0001
contact = contact_R * np.ones((1, n))
R = np.vstack(( contact, R, contact ))
m = m + 2

Adim =(n-1)*(m-1) + 1
#A = lil_matrix((Adim, Adim))
row = []
col = []
element = []
def add_element(r, c, elem):
    row.append(r)
    col.append(c)
    element.append(elem)

Rshape = np.shape(R)

# Fill Enrique array
for i in range(m-1):
    for j in range(n-1):
        # Loop numbers
        k = i * (n - 1) + j
        k_above = k - (n - 1)
        k_below = k + (n - 1)
        k_left = k - 1
        k_right = k + 1

        # End up doing this many times, better to figure it out beforehand
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
            #add_element(k, Adim - 1, R_left + R_above)
        elif top and right:
            add_element(k, k, R_above + R_right + R_left + R_below)
            add_element(k, k_below, -R_below)
            add_element(k, k_left, -R_left)
            #A[k, Adim - 1] = R_right + R_above
        elif bottom and left:
            add_element(k, k, R_above + R_right + R_left + R_below)
            add_element(k, k_above, -R_above)
            add_element(k, k_right, -R_right)
            add_element(k, Adim-1, -R_left)
            #A[k, Adim - 1] = R_left + R_below
        elif bottom and right:
            add_element(k, k, R_above + R_right + R_left + R_below)
            add_element(k, k_above, -R_above)
            add_element(k, k_left, -R_left)
            #A[k, Adim - 1] = R_right + R_below
        elif top:
            add_element(k, k, R_above + R_right + R_left + R_below)
            add_element(k, k_below, -R_below)
            add_element(k, k_left, -R_left)
            add_element(k, k_right, -R_right)
            #A[k, Adim - 1] = R_above
        elif bottom:
            add_element(k, k, R_above + R_right + R_left + R_below)
            add_element(k, k_above, -R_above)
            add_element(k, k_left, -R_left)
            add_element(k, k_right, -R_right)
            #A[k, Adim - 1] = R_below
        elif left:
            add_element(k, k, R_above + R_right + R_left + R_below)
            add_element(k, k_above, -R_above)
            add_element(k, k_below, -R_below)
            add_element(k, k_right, -R_right)
            add_element(k, Adim-1, -R_left)
            #A[k, Adim - 1] = R_left
        elif right:
            add_element(k, k, R_above + R_right + R_left + R_below)
            add_element(k, k_above, -R_above)
            add_element(k, k_below, -R_below)
            add_element(k, k_left, -R_left)
            #A[k, Adim - 1] = R_right
        else:
            add_element(k, k, R_above + R_right + R_left + R_below)
            add_element(k, k_above, -R_above)
            add_element(k, k_below, -R_below)
            add_element(k, k_left, -R_left)
            add_element(k, k_right, -R_right)


#         if top and left:
#             A[k, k] = R_above + R_right + R_left + R_below
#             A[k, k_below] = -R_below
#             A[k, k_right] = -R_right
#             A[k, -1] = -R_left
#             #A[k, Adim - 1] = R_left + R_above
#         elif top and right:
#             A[k, k] = R_above + R_right + R_left + R_below
#             A[k, k_below] = -R_below
#             A[k, k_left] = -R_left
#             #A[k, Adim - 1] = R_right + R_above
#         elif bottom and left:
#             A[k, k] = R_above + R_right + R_left + R_below
#             A[k, k_above] = -R_above
#             A[k, k_right] = -R_right
#             A[k, -1] = -R_left
#             #A[k, Adim - 1] = R_left + R_below
#         elif bottom and right:
#             A[k, k] = R_above + R_right + R_left + R_below
#             A[k, k_above] = -R_above
#             A[k, k_left] = -R_left
#             #A[k, Adim - 1] = R_right + R_below
#         elif top:
#             A[k, k] = R_above + R_right + R_left + R_below
#             A[k, k_below] = -R_below
#             A[k, k_left] = -R_left
#             A[k, k_right] = -R_right
#             #A[k, Adim - 1] = R_above
#         elif bottom:
#             A[k, k] = R_above + R_right + R_left + R_below
#             A[k, k_above] = -R_above
#             A[k, k_left] = -R_left
#             A[k, k_right] = -R_right
#             #A[k, Adim - 1] = R_below
#         elif left:
#             A[k, k] = R_above + R_right + R_left + R_below
#             A[k, k_above] = -R_above
#             A[k, k_below] = -R_below
#             A[k, k_right] = -R_right
#             A[k, -1] = -R_left
#             #A[k, Adim - 1] = R_left
#         elif right:
#             A[k, k] = R_above + R_right + R_left + R_below
#             A[k, k_above] = -R_above
#             A[k, k_below] = -R_below
#             A[k, k_left] = -R_left
#             #A[k, Adim - 1] = R_right
#         else:
#             A[k, k] = R_above + R_right + R_left + R_below
#             A[k, k_above] = -R_above
#             A[k, k_below] = -R_below
#             A[k, k_left] = -R_left
#             A[k, k_right] = -R_right

# Put 1V across top and bottom
for i in range(m-1):
    R_edge = (R[i, 0] + R[i+1, 0]) / 2
    k = i * (n - 1)
    #A[Adim-1, k] = R_edge
    add_element(Adim-1, k, -R_edge)
    add_element(Adim-1, Adim-1, R_edge)
b = np.zeros(Adim)
b[-1] = 1

A = coo_matrix((element, (row, col)), shape=(Adim, Adim))

t1 = time.time()

print('Seconds to generate Enrique matrix: {}'.format(t1 - t0))

A = A.tocsr()
x = spsolve(A, b)

t2 = time.time()

print('Loop current computation took {} seconds'.format(t2 - t1))

# Calculate voltages from current
I_loop = x[:-1].reshape((m-1, n-1))
I_vert = np.hstack((x[-1]*np.ones((m-1, 1)), I_loop)) - np.hstack((I_loop, np.zeros((m-1, 1))))
R_vert = R[:-1, :] / 2. + R[1:, :] / 2.
V = np.vstack((np.zeros((1, n)), cumsum(I_vert * R_vert, axis=0)))
plt.figure()
imshow(V, interpolation='none')
plt.show()


# fig, ax = plt.subplots()
# ax.imshow(V, interpolation='none')

# Electric field
E = np.gradient(V)
Ey = E[0]
Ex = E[1]
Emag = [sqrt(x**2 + y**2) for x,y in zip(*E)]
fig, ax = plt.subplots()
ax.imshow(Emag, interpolation='none')

# Currents
# y = -1 * CA * matrix(x).T


