# Poisson

We solve the following [PDE]([Weak adversarial networks for high-dimensional partial differential equations - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0021999120301832))

$$
\frac{\partial u}{\partial t}-\frac{\pi^2}{4}\sum_{i=1}^{d}\sin(\frac{\pi}{2}x_i)-\Delta_x u(x,t)=0,t\in[s,T],x\in D⊂\mathbb{R}^d
$$

whose boundary condition is given by

$$
u(x,t)=q(x,t)=\sum_{i=1}^d\sin(\frac\pi2 x_i),t\in[s,T],x\in \partial D\sub \mathbb{R}^{d-1}
$$
The nonlinear term is given by
$$
f(x,t,u,z)=-\frac{\pi^2}{4}\sum_{i=1}^{d}\sin(\frac{\pi}{2}x_i)
$$


This PDE has an explicit solution at time $t$
$$
u(x,t)=\sum_{i=1}^d\sin(\frac\pi2 x_i).
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

