

# Linear_Convection_Diffusion

Consider the following equation
$$
\frac{\partial u}{\partial t}-\frac1d div_x u+\Delta_x u=0,t\in[s,T],x\in D\sub\mathbb{R}^d
$$
whose terminal condition is given by
$$
u(x,T)=g(x):=\sum_{i=1}^d x_i+T
$$
without any boundary constraints.

The nonlinear term is given by
$$
F(u,z)(x,t)=0
$$
and the explicit solution is
$$
u(x,t)=\sum_{i=1}^d x_i+t
$$
which is our target in this section.

## Parameters

Specifically, we consider the problem for
$$
d=10, 20, 30, 40, \mu=-\frac1d,\sigma=\sqrt{2}, D=[0,0.5]^{10,20,30,60}, s=0, T=0.5
$$

