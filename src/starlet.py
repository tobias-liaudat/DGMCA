# Module that implements in 2D starlet transform

################# Useful codes

def length(x=0):

    import numpy as np
    l = np.max(np.shape(x))
    return l

################# EXP MAP

    def Exp(self,x,y):
        from cmath import exp
        z=[]
        if len(x)==len(y):
            i=0
            while i<len(x):
                z.append(x[i]*exp(1j*y[i]))
                i+=1
        elif len(x)==1:
            i=0
            while i<len(y):
                z.append(x*exp(1j*y[i]))
                i+=1
        return z

    def Log(self,x,y):
        from cmath import phase
        z=[]
        if len(x)==len(y):
            i=0
            while i<len(x):
                z.append(phase(y[i])-phase(x[i]))
                i+=1
        elif len(x)==1:
            i=0
            while i<len(y):
                z.append(phase(y[i])-phase(x[0]))
                i+=1
        return z
    def norm(self,x):
        return np.abs(x)


################# 1D convolution

def filter_1d(xin=0,h=0,boption=3):

    import numpy as np
    import scipy.linalg as lng
    import copy as cp

    x = np.squeeze(cp.copy(xin));
    n = length(x);
    m = length(h);
    y = cp.copy(x);
    m2 = np.int(m/2.)
    z = np.zeros(1,m);

    for r in range(np.int(m2)):

        if boption == 1: # --- zero padding

            z = np.concatenate([np.zeros(m-r-m2-1),x[0:r+m2+1]],axis=0)

        if boption == 2: # --- periodicity

            z = np.concatenate([x[n-(m-(r+m2))+1:n],x[0:r+m2+1]],axis=0)

        if boption == 3: # --- mirror

            u = x[0:m-(r+m2)-1];
            u = u[::-1]
            z = np.concatenate([u,x[0:r+m2+1]],axis=0)

        y[r] = np.sum(z*h)

    a = np.arange(np.int(m2),np.int(n-m+m2),1)

    for r in a:

        y[r] = np.sum(h*x[r-m2:m+r-m2])


    a = np.arange(np.int(n-m+m2+1),n,1)

    for r in a:

        if boption == 1: # --- zero padding

            z = np.concatenate([x[r-m2:n],np.zeros(m - (n-r) - m2)],axis=0)

        if boption == 2: # --- periodicity

            z = np.concatenate([x[r-m2:n],x[0:m - (n-r) - m2]],axis=0)

        if boption == 3: # --- mirror

            u = x[n - (m - (n-r) - m2 -1)-1:n]
            u = u[::-1]
            z = np.concatenate([x[r-m2:n],u],axis=0)

        y[r] = np.sum(z*h)

    return y

################# 1D convolution with the "a trous" algorithm

def Apply_H1(x=0,h=0,scale=1,boption=3):

	import numpy as np
	import copy as cp

	m = length(h)

	if scale > 1:
		p = (m-1)*np.power(2,(scale-1)) + 1
		g = np.zeros( p)
		z = np.linspace(0,m-1,m)*np.power(2,(scale-1))
		g[z.astype(int)] = h

	else:
		g = h

	y = filter_1d(x,g,boption)

	return y


################# 2D "a trous" algorithm

def Starlet_Forward(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):

	import numpy as np
	import copy as cp

	nx = np.shape(x)
	c = np.zeros((nx[0],nx[1]))
	w = np.zeros((nx[0],nx[1],J))

	c = cp.copy(x)
	cnew = cp.copy(x)

	for scale in range(J):

		for r in range(nx[0]):

			cnew[r,:] = Apply_H1(c[r,:],h,scale,boption)

		for r in range(nx[1]):

			cnew[:,r] = Apply_H1(cnew[:,r],h,scale,boption)

		w[:,:,scale] = c - cnew;

		c = cp.copy(cnew);

	return c,w

################# 2D "a trous" algorithm

def Starlet_Inverse(c=0,w=0):

	import numpy as np

	x = c+np.sum(w,axis=2)

	return x
