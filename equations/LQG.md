# LQG

We solve the following [PDE](https://www.pnas.org/doi/epdf/10.1073/pnas.1718942115)
$$
\begin{align}

 &\frac{\partial u}{\partial t}-\lambda\|\nabla_x u(x,t)\|^2+\Delta_x u(x,t)=0,\\&t\in[s,T],x\in D⊂\mathbb{R}^d

\end{align}
$$
whose terminal condition is given by

$$
\begin{align}

u(x,T)=g(x):=\ln (\frac{1+\|x\|^2}{2}),

\end{align}
$$
without any boundary constraints.



This PDE has an simulation solution at time $t$
$$
\begin{align}

u(x,t)=-\frac1\lambda\ln\left(\mathbb{E}\left[\exp\left(-\lambda g(x+\sqrt{2}W_{T-t})\right)\right]\right).

\end{align}
$$
which is our target in this section.



Specifically, we consider the problem for
$$
\begin{align}

d&=100\\σ&=0.25\\

D&=[-0.5,0.5]^{100}\\
\lambda&=1\\
s&=0\\T&=0.5.

\end{align}
$$
