import numpy as np
import matplotlib.pyplot as plt

class NormalShperes(object):
    '''Normal spheres in high dimensions'''
    def __init__(self, dim, n_spheres, r_spheres):
        #initialize the normal spheres
        self.dim=dim
        self.n_spheres=n_spheres
        self.r_spheres=r_spheres
    def sample(self, n_samples):
        #sample from the normal spheres
        x=np.random.normal(size=(n_samples,self.dim))
        for i in range(self.n_spheres):
            x+=np.random.normal(size=(n_samples,1))*self.r_spheres[i]
        return x
    def plot(self, n_samples):
        #plot the normal spheres
        x=self.sample(n_samples)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:,0], x[:,1], x[:,2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()