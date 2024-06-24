# Explicit_Solution_Example_Rescale

We solve the following [PDE](https://arxiv.org/abs/1708.03223)(an adapted version):

$$
\frac{\partial u}{\partial t}+(\sigma^2 d u(x,t)-1 -\frac{d\sigma^2}{2})div_x u(x,t)+\frac {\sigma^2 d^2}2 \Delta_x u(x,t)=0,t\in[s,T],x\in D \in \mathbb{R}^d
$$

whose terminal condition is given by

$$
u(x,T)=g(x):=\frac{\exp(T+\frac{1}{d}\sum\limits_{i=1}^d x_i)}{1+\exp(T+\frac{1}{d}\sum\limits_{i=1}^d x_i)}
$$

without any boundary constraints.



This PDE has an explicit solution at time $t$:

$$
u(x,t)=\frac{\exp(t+\frac{1}{d}\sum\limits_{i=1}^d x_i)}{1+\exp(t+\frac{1}{d}\sum\limits_{i=1}^d x_i)}
$$

which is our target in this section.



Specifically, we consider the problem for

$$
d=100, \sigma=0.25, D=[-0.5,0.5]^{100}, s=0, T=0.5
$$

