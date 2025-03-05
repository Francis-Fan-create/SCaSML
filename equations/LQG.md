

# LQG

Consider the following equation
$$
\frac{\partial u}{\partial t}+\Delta_x u-\|\nabla_x u\|^2=0,t\in[s,T],x\in D\sub\mathbb{R}^d
$$
whose terminal condition is given by
$$
u(x,T)=g(x):=\log(\frac{1+\|x\|^2}{2})
$$
without any boundary constraints.

The nonlinear term is given by
$$
F(u,z)(x,t)=-\frac{\|z\|^2}{2}
$$
and the explicit solution is
$$
u(x,t)=-\log(\mathbb{E}\exp(-\log(\frac{1+\|x+\sqrt2W_{T-t}\|^2}{2})))=-\log(\mathbb{E}\frac{2}{1+\|x+\sqrt2W_{T-t}\|^2})
$$
which is our target in this section.

## Parameters

Specifically, we consider the problem for
$$
d=20, 40, 60, 80, \mu=0,\sigma=\sqrt{2}, D=[0,0.5]^{20,40,60,80}, s=0, T=0.5
$$

