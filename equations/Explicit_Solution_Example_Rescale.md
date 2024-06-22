# Explicit_Solution_Example_Rescale

We solve the following [PDE](https://arxiv.org/abs/1708.03223)(an adapted version)
$$
\begin{align}

 &\frac{\partial u}{\partial t}+(\sigma^2 d u(x,t)-1 -\frac{\sigma^2 d}{2})div_x u(x,t)+\frac {\sigma^2 d^2 }2 \Delta_x u(x,t)=0,\\&t\in[s,T],x\in D⊂\mathbb{R}^d

\end{align}
$$
whose terminal condition is given by
$$
\begin{align}

u(x,0)=g(x):=\frac{\exp(T+\frac{1}{d}\sum^d_{i=1}x_i)}{1+\exp(T+\frac{1}{d}\sum^d_{i=1}x_i)},

\end{align}
$$
without any boundary constraints.



This PDE has an explicit solution at time $t$
$$
\begin{align}

u(x,t)=\frac{\exp(t+\frac{1}{d}\sum^d_{i=1}x_i)}{1+\exp(t+\frac{1}{d}\sum^d_{i=1}x_i)},

\end{align}
$$
which is our target in this section.



Specifically, we consider the problem for
$$
\begin{align}

d&=100\\σ&=0.25\\

D&=[-0.5,0.5]^{500}\\

s&=0\\T&=0.5.

\end{align}
$$
