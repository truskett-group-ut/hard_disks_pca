from numpy import arctan, mean, var
import pyvoro

def BOP(frames):
    N = len(frame['coords'][0])
    magnitudes = []
    phases = []
    Rs = []
    Is = []
    #loop over every frame
    for frame in frames:
        voro = pyvoro.compute_2d_voronoi(
                                          frame['coords'], # point positions
                                          [[0.0, frame['L']], [0.0, frame['L']]], # limits
                                          4.0, # block size
                                          periodic=[True, True]
                                        )
        voro = array(voro)
    
        #loop over the cells and use nearest neighbor details to calculate BOP
        R_tot, I_tot = 0.0, 0.0
        for i in range(len(voro)):
            ri = voro[i]['original']
            nbrs = voro[i]['faces']
            n = len(nbrs)
            R, I = 0.0, 0.0
            for nbr in nbrs:
                j = nbr['adjacent_cell']
                rj = voro[j]['original']
                rij = ri - rj
                rij = rij - rint(rij/frame['L'])*frame['L']
                c = rij[0]/norm(rij)
                s = rij[1]/norm(rij)
                Rl = (c**6) - 15.0*(c**4)*(s**2) + 15.0*(c**2)*(s**4) - (s**6)
                Il = 6.0*(c**5)*s - 20.0*(c**3)*(s**3) + 6.0*c*(s**5)
                R = R + Rl
                I = I + Il
            #single particle average of the local order parameter
            R = R/float(n)
            I = I/float(n)
       
            #calculate the whole frame real and imaginary components
            R_tot = R_tot + R
            I_tot = I_tot + I
        
        Rs.append(R_tot)
        Is.append(I_tot)
    
    #global system averages
    R_avg, I_avg = mean(Rs), mean(Is)
    Psi_avg = sqrt(R_avg*R_avg + I_avg*I_avg)
    R_var, I_var = var(Rs), var(Is)
    Psi_var = R_var + I_var
    
    return (Psi_avg/float(N), Psi_var/float(N))           
    