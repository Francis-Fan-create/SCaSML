

# LQG

Consider the following equation
$$
\frac{\partial u}{\partial t}+\Delta_x u-\|\nabla_x u\|^2=0,t\in[s,T],x\in D\sub\mathbb{R}^d
$$
whose terminal condition is given by
$$
u(x,T)=g(x):=\log\left(\frac{1+\sum_{i=1}^{d-1}\left[c_{1,i}(x_i-x_{i+1})^2+c_{2,i}x_{i+1}^2\right]}{2}\right), c_{1,i},c_{2,i}\sim \text{Unif}[0.5,1.5]
$$
without any boundary constraints.

The nonlinear term is given by
$$
F(u,z)(x,t)=-\frac{\|z\|^2}{2}
$$
and the explicit solution is
$$
u(x,t)=-\log(\mathbb{E}\exp\left(-g(x+\sqrt2W_{T-t} )\right))
$$
which is our target in this section.

## Parameters

Specifically, we consider the problem for
$$
d=100,120, 140, 160, \mu=0,\sigma=\sqrt{2}, D=\mathbb{B}^{100,120,140,160}, s=0, T=1
$$

