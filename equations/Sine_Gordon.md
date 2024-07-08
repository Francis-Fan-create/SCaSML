# Sine_Gordon

We solve the following [PDE](https://arxiv.org/pdf/2005.10206):

$$
\frac{\partial u}{\partial t}+\sin u(x,t)+\Delta_x u(x,t)=0,t\in[s,T],x\in D \in \mathbb{R}^d
$$

whose terminal condition is given by

$$
u(x,T)=g(x):=\frac{1}{2+0.4\|x\|_2^2}
$$

without any boundary constraints.



The nonlinear term is given by
$$
f(x,t,u,z)=\sin u
$$


This PDE does not have an explicit solution at time $t$.



Specifically, we consider the problem for

$$
d=100, \mu=0,\sigma=\sqrt 2, D=[-0.5,0.5]^{100}, s=0, T=1
$$

and

$$
d=250,\mu=0, \sigma=\sqrt 2, D=[-0.5,0.5]^{250}, s=0, T=1
$$
