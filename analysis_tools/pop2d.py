from numpy import array, sum, dot, transpose, power, cos, sin, exp

def POP2D(frames, k, theta):
    #unit vector, particle number
    k_v = k*array([cos(theta), sin(theta)])
    N = len(frames[0]['coords'])
    M = len(frames)
    
    #calculate the traditional positional OP used for hard disks 
    pop = 0.0
    for frame in frames:
        R_v = frame['coords']
        dot_prod = dot(R_v, transpose(k_v))
        pop = pop + power(abs(sum(exp(1j*dot_prod))/N), 2)/M
        
    return pop