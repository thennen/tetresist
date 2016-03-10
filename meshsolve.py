import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
# TODO:  Find a way to generate matrices more quickly by storing previous one
# and making modifications
# also make another version which wraps in the horizontal direction

def solve(R, V_contact=1):
    ''' Compute currents and voltage of mesh of resistors. Top electrode at 0V '''
    out = dict()

    m, n = np.shape(R)

    # Put some contacts on top and bottom
    contact_R = 0
    contact = contact_R * np.ones((1, n))
    R = np.vstack(( contact, R, contact ))
    m = m + 2

    Adim =(n-1)*(m-1) + 1

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

    A = coo_matrix((element, (row, col)), shape=(Adim, Adim))
    A = A.tocsr()
    x = spsolve(A, b)

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

    # Electric field
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


    return out
