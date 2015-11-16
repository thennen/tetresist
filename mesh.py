DIFF_THRESHOLD = 1e-10
 
class Fixed:
    FREE = 0
    A = 1
    B = 2
 
class Node:
    __slots__ = ["voltage", "fixed"]
    def __init__(self, v=0.0, f=Fixed.FREE):
        self.voltage = v
        self.fixed = f

def set_boundary(m):
    ''' Top row +1V, bottom row -0V '''
    h = len(m)
    w = len(m[0])
    m[0] = [Node(1.0, Fixed.A)]*w
    m[-1] = [Node(0.0, Fixed.B)]*w
 
def calc_difference(m, d, R):
    h = len(m)
    w = len(m[0])
    total = 0.0
 
    for i in xrange(h):
        for j in xrange(w):
            v = 0.0
            n = 0.0
            if i != 0:
                a = 2/(R[i-1][j] + R[i][j])
                v += m[i-1][j].voltage*a
                n += a
            if j != 0:
                a = 2/(R[i][j-1] + R[i][j])
                v += m[i][j-1].voltage*a
                n += a
            if i < h-1:
                a = 2/(R[i+1][j] + R[i][j])
                v += m[i+1][j].voltage*a
                n += a
            if j < w-1:
                a = 2/(R[i][j+1] + R[i][j])
                v += m[i][j+1].voltage*a
                n += a
            v = m[i][j].voltage - v / n
 
            d[i][j].voltage = v
            if m[i][j].fixed == Fixed.FREE:
                total += v ** 2
    return total
 
def iter(m, R):

    h = len(m)
    w = len(m[0])
    difference = [[Node() for j in xrange(w)] for i in xrange(h)]
 
    while True:
        set_boundary(m) # Enforce boundary conditions.
        if calc_difference(m, difference, R) < DIFF_THRESHOLD:
            break
        for i, di in enumerate(difference):
            for j, dij in enumerate(di):
                m[i][j].voltage -= dij.voltage
 
    # This is no longer generally true
    # Count current leaving top electrode
    # return sum([d.voltage/r for d,r in zip(difference[0], R[0])])

    # calculate currents and power
    I = [[[0,0] for j in xrange(w)] for i in xrange(h)]
    P = [[0 for j in xrange(w)] for i in xrange(h)]
    Icontact = 0

    for i, di in enumerate(difference):
        for j, dij in enumerate(di):
            if i != 0:
                a = 2/(R[i-1][j] + R[i][j])
                v = m[i][j].voltage - m[i-1][j].voltage
                cur = a*v
                P[i][j] += cur**2 * R[i][j]
                I[i][j][0] += cur 
            else:
                # keep track of current leaving contact
                a = 2/(R[i+1][j] + R[i][j])
                Icontact += a*dij.voltage
            if j != 0:
                a = 2/(R[i][j-1] + R[i][j])
                v = m[i][j].voltage - m[i][j-1].voltage
                cur = a*v
                P[i][j] += cur**2 * R[i][j]
                I[i][j][1] += cur 
            if i < h-1:
                a = 2/(R[i+1][j] + R[i][j])
                v = m[i][j].voltage - m[i+1][j].voltage
                cur = a*v
                P[i][j] += cur**2 * R[i][j]
                I[i][j][0] -= cur 
            if j < w-1:
                a = 2/(R[i][j+1] + R[i][j])
                v = m[i][j].voltage - m[i][j+1].voltage
                cur = a*v
                P[i][j] += cur**2 * R[i][j]
                I[i][j][1] -= cur 

    # returning everything per volt (conductance), P/V^2
    return Icontact, I, P 

def newmesh(h=20, w=10):
    return [[Node() for j in xrange(w)] for i in xrange(h)]
