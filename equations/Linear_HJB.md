# Linear_HJB

We solve the following [PDE](https://arxiv.org/abs/2206.02016)(labeled as Eq(100) in the original paper)

$$
\frac{\partial u}{\partial t}-\frac1d div_x u(x,t)+2+\Delta_x u(x,t)=0,t\in[s,T],x\in D⊂\mathbb{R}^d
$$

whose terminal condition is given by


$$
u(x,T)=g(x):=\sum_{i=1}^d x_i,
$$

without any boundary constraints.

The nonlinear term is given by
$$
F(u,z)(x,t)=2
$$


This PDE has an explicit solution at time $t$
$$
u(x,t)=\sum_{i=1}^d x_i+(T-t).
$$

which is our target in this section.



Specifically, we consider the problem for

$$
d=20, 40, 60, 80,\mu=-1/d, \sigma=\sqrt{2}, D=[-0.5,0.5]^{20,40,60,80}, s=0, T=0.5
$$

