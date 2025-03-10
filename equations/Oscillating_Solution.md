# Oscillating_Solution

Consider the following equation
$$
\frac{\partial u}{\partial t}+\frac12\Delta_x u+\min\{1,(u-u^*)^2\}=0,t\in[s,T],x\in D\sub\mathbb{R}^d
$$
whose terminal condition is given by
$$
u^\star(x,T)=g(x):=1.6+\sin\left(0.1\sum_{i=1}^dx_i\right)
$$
without any boundary constraints.

The nonlinear term is given by
$$
F(u,z)(x,t)=\min\{1,(u-u^*)^2\}
$$
and the explicit solution is
$$
u^\star(x,t)=1.6+\sin\left(0.1\sum_{i=1}^dx_i\right)\exp\left(\frac{0.01d(t-T)}{2}\right)
$$
which is our target in this section.

## Parameters

Specifically, we consider the problem for
$$
d=100,120, 140, 160, \mu=0,\sigma=\sqrt{2}, D=\mathbb{B}^{100,120,140,160}, s=0, T=1
$$

