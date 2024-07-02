# Allen_Cahn

We solve the following [PDE](https://arxiv.org/pdf/2005.10206):

$$
\frac{\partial u}{\partial t}+[u(x,t)-u(x,t)^3]+\Delta_x u(x,t)=0,t\in[s,T],x\in D \in \mathbb{R}^d
$$

whose terminal condition is given by

$$
u(x,T)=g(x):=\frac{1}{2+0.4\|x\|_2^2}
$$

without any boundary constraints.



This PDE do not has an explicit solution at time $t$.



Specifically, we consider the problem for

$$
d=100, \mu=0,\sigma=0.25, D=[-0.5,0.5]^{100}, s=0, T=1
$$

and

$$
d=250,\mu=0, \sigma=0.25, D=[-0.5,0.5]^{250}, s=0, T=1
$$
