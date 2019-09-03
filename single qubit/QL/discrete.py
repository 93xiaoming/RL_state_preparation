



def phase2(z):
    '''
    return phase angle in [0, 2pi]
    '''
    phase = cmath.phase(z)
    if phase < 0:
        phase += 2*math.pi
    return phase

# mesh grid of Bloch sphere
# Coversion from Bloch state to Q-table state
# theta: 60, phi: 60
# theta_i = round(theta/(pi/60)), phi_i = round(phi/(pi/60))
# state: k --> (smaples_phi) * theta_i + phi_i
# Q_{k,a}, k: 0-?, a: {0,1}.
    

def state_to_lattice_point(state): # input quantum state psi, output the descrete state closest to it 
    Dtheta = np.pi/30
    Dphi = np.pi/30
    '''
    Note: phi = 0 or 2pi are the same
    return the list [theta_i, phi_i]
    '''
    if state[0,0] == 0:
        ## Special case 1: [0, 1]
        theta, phi = math.pi, 0
    else:
        conj = state[0,0].conj()
        state_reg = state * (conj/abs(conj))
        # print(state_reg[0,0].real)
        if (state_reg[0,0].real)>= 1:
            # Unitary should preserve norm
            theta, phi = 0, 0
        else: 
            # print(state_reg[0,0].imag)                  # this should be 0
            theta = 2 * math.acos(state_reg[0,0].real)
            # state_reg[1,0]/sin(theta/2) = cos(pi) + i sin(pi)
            if theta == 0:
                ## Special case 2: [1, 0]
                phi = 0
            else:
                phi = phase2(state_reg[1,0]/math.sin(theta/2))  #force the phase of the first elements to be 0.
    theta_i = round(theta/Dtheta)
    phi_i = round(phi/Dphi)
    if phi_i == round(2*math.pi/Dphi):
        phi_i = 0
    return [theta_i, phi_i]


#can try :
#state = np.array([[1],[0]], dtype=complex)
#state = np.array([[np.cos(0.1)],[np.sin(0.1)]], dtype=complex)
#x=state_to_lattice_point(state)  
#print(x)
        














