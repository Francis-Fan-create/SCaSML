# Neumann_Boundary

We solve the following [PDE]([Weak adversarial networks for high-dimensional partial differential equations - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0021999120301832))

$$
\frac{\partial u}{\partial t}+[2u-(\frac{\pi^2}{2}+2)\sin(\frac{\pi}{2}x_1)\cos(\frac{\pi}{2}x_2)]-\Delta_x u(x,t)=0,t\in[s,T],x\in D⊂\mathbb{R}^d
$$

whose Neumann boundary condition is given by

$$
\frac{\partial u}{\partial \vec{n}}(x,t)=q(x,t):=[\frac{\pi}{2}\cos\left(\frac{\pi}{2}x_{1}\right)\cos\left(\frac{\pi}{2}x_{2}\right),-\frac{\pi}{2}\sin\left(\frac{\pi}{2}x_{1}\right)\sin\left(\frac{\pi}{2}x_{2}\right),0,\cdots,0]\cdot\vec{n},t\in[s,T],x\in \partial D\sub \mathbb{R}^{d-1}
$$
The nonlinear term is given by
$$
f(x,t,u,z)=2u-(\frac{\pi^2}{2}+2)\sin(\frac{\pi}{2}x_1)\cos(\frac{\pi}{2}x_2)
$$


This PDE has an explicit solution at time $t$
$$
u(x,t)=\sin(\frac{\pi}{2}x_1)\cos(\frac{\pi}{2}x_2).
$$

which is our target in this section.



Specifically, we consider the problem for

$$
d=100, \mu=0,\sigma=-\sqrt{2}, D=[-0.5,0.5]^{100}, s=0, T=0.5
$$

and

$$
d=250,\mu=0, \sigma=-\sqrt{2}, D=[-0.5,0.5]^{250}, s=0, T=0.5
$$

