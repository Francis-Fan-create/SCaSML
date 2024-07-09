# Diffusion_Reaction

We solve the following [PDE]([Weak adversarial networks for high-dimensional partial differential equations - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0021999120301832))

$$
\frac{\partial u}{\partial t}+[-u^2-(\frac{\pi^2}{2}-2)\sin(\frac{\pi}{2}x_1)e^{-t}+4\sin^2(\frac{\pi}{2}x_1)e^{-2t}]-\Delta_x u(x,t)=0,t\in[s,T],x\in D⊂\mathbb{R}^d
$$

whose initial condition is given by


$$
u(x,0)=h(x):=2\sin(\frac{\pi}{2}x_1),
$$

and boundary conditions is given by
$$
u(x,t)=q(x,t):=2\sin(\frac{\pi}{2}x_1)e^{-t},t\in[s,T],x\in\partial D\sub \mathbb{R}^{d-1}.
$$
The nonlinear term is given by
$$
f(x,t,u,z)=-u^2-(\frac{\pi^2}{2}-2)\sin(\frac{\pi}{2}x_1)e^{-t}+4\sin^2(\frac{\pi}{2}x_1)e^{-2t}
$$


This PDE has an explicit solution at time $t$
$$
u(x,t)=2\sin(\frac{\pi}{2}x_1)e^{-t}.
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

