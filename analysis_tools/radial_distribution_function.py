from numpy import histogram, pi, power, rint
from numpy.linalg import norm
from numpy import trapz, arange

#function for calculating the rdf
def RDF(frames, dr):
    M = len(frames)
    N = float(len(frames[0]['coords']))
    L = frames[0]['L']
    D = frames[0]['D']
    if D not in [2, 3]:
        raise Exception('RDF code can only handle 2 and 3 dimensions right now. Sorry.')
    rho = float((N-1))/power(L, D)
    r_edg = arange(0.0, L/2.0, dr)
    r = (r_edg[:-1] + r_edg[1:])/2.0
    hist = 0.0*r
    
    #loop over the frames
    for frame in frames:
        coords = frame['coords']

        #loop over each particle
        for coord in coords:
            #nearest neighbor coordinate wrapping
            Rpj = coord - coords
            Rpj = Rpj - rint(Rpj/L)*L
            Rpj = norm(Rpj, axis=1)
            
            #calculate the histogram
            hist = hist + histogram(Rpj, bins=r_edg)[0]
    
    #normalize out the number of frames and 
    hist = hist/float(M*(N-1))
    gr = hist/((2.0*float(D-1)*pi*power(r, D-1)*dr)*rho)
    
    return r, gr

#function for calculating the integral of h(r)
def PositionalSuceptibility(frames, dr):
    N = float(len(frames[0]['coords']))
    L = frames[0]['L']
    D = frames[0]['D']
    if D not in [2, 3]:
        raise Exception('Positional susceptibility code can only handle 2 and 3 dimensions right now. Sorry.')
    rho = float(N)/power(L, D)
      
    #compute the rdfs and calculate susceptibility
    r, gr = RDF(frames, dr)
    pos_spt = 2.0*float(D-1)*pi*rho*trapz(power(r, D-1)*abs(gr-1.0))
    
    return (pos_spt, r, gr)