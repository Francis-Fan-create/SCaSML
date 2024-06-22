# Complicated_HJB

We solve the following [PDE](https://arxiv.org/abs/2206.02016)(labeled as Eq(100) in the original paper)
$$
\begin{align}

 &\frac{\partial u}{\partial t}-(\frac1d\sum_{i=1}^d|\frac{\partial u}{\partial x_i}|^c+2)+\Delta_x u(x,t)=0,\\&t\in[s,T],x\in D⊂\mathbb{R}^d

\end{align}
$$
whose terminal condition is given by

$$
\begin{align}

u(x,T)=g(x):=\sum_{i=1}^d x_i,

\end{align}
$$
without any boundary constraints.



This PDE has an explicit solution at time $t$
$$
\begin{align}

u(x,t)=\sum_{i=1}^d x_i+(T-t).

\end{align}
$$
which is our target in this section.



Specifically, we consider the problem for
$$
\begin{align}

d&=100\\σ&=0.25\\

D&=[-0.5,0.5]^{100}\\

s&=0\\T&=0.5.

\end{align}
$$
