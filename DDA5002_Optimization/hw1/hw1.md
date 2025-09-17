<p align='center'><b>DDA 5002 Optimization Fall 2025
<p align='center'>Homework # 1
<p align='center'>Student ID: 225040065

##### Question 1

(a)

$x_1: \text{number of the product of the first type produced per day}$

$x_2: \text{number of the product of the second type produced per day}$

$\text{objective function:}$
$$
\max(7.8x_1 + 7.1x_2)
$$
$\text{constraints:}$
$$
\frac{1}{4}x_1 + \frac{1}{3}x_2 \le 90 \\
\frac{1}{8}x_1 + \frac{1}{3}x_2 \le 80 \\
x_1, x_2 \ge 0
$$

(b)

(i) **can** be easily incorporated into the linear program formulation, while (ii) **cannot**. 

**for (i) modification (Linear Program Formulation)**

$x_1: \text{number of the product of the first type produced per day}$

$x_2: \text{number of the product of the second type produced per day}$

$h: \text{number of hours of overtime assembly labor per day}$

$\text{objective function:}$
$$
\max(7.8x_1 + 7.1x_2 - 7h) \\
\begin{align}
s.t. \quad \frac{1}{4}x_1 + \frac{1}{3}x_2 &\le 90 + h \\
\frac{1}{8}x_1 + \frac{1}{3}x_2 &\le 80 \\
h &\le 50 \\
x_1, x_2, h &\ge 0
\end{align}
$$
**for (ii) modification**

$x_1: \text{number of the product of the first type produced per day}$

$x_2: \text{number of the product of the second type produced per day}$

$\text{objective function:}$
$$
\max(7.8x_1 + 7.1x_2 - \mathbb{1}\{7.8x_1 + 7.1x_2 > 300\}(0.12x_1 + 0.09x_2))
$$
$\text{constraints:}$
$$
\frac{1}{4}x_1 + \frac{1}{3}x_2 \le 90 \\
\frac{1}{8}x_1 + \frac{1}{3}x_2 \le 80 \\
x_1, x_2\ge 0
$$



##### Question 2

(a)

$p_k: \text{daily power consume of the k-th car, and we have: } p_1 = 10, p_2 = 8, p_3 = 13, p_4 = 15, p_5 = 9$

$d_k: \text{total mileage traveled of the k-th car, and we have: } d_1 = 60, d_2 = 55, d_3 = 75, d_4 = 80, d_5 = 64$

$m, b: \text{linear program parameters}$

$\text{objective function:}$
$$
\min(\Sigma^{5}_{k=1}y_k)
$$
$\text{constraints:}$
$$
- y_k \le d_k - (mp_k + b) \le y_k,\quad k = 1, 2, 3, 4, 5 \\
m, b\ge 0
$$

(b)

codes:



report:



##### Question 3

$\text{period: } t = 1, 2,\cdots, T$

$\text{plant: } p = 1, 2,\cdots, P$

$\text{outlet: } o = 1, 2,\cdots, O$

$x_{t,p}: \text{number of plant } p \text{ manufatured in period } t$

$m_{t,p}: \text{cost of manufaturing plant } p \text{ in period } t$

$y_{t,p,o}: \text{number of plant } p \text{ shipped to outlet } o \text{ in period } t$

$c_{p,o}: \text{cost of shipping plant } p \text{ to outlet } o \text{ in period } t$

$z_{t,o}: \text{number of plant sold to outlet } o \text{ in period } t$

$r_{t,o}: \text{price of selling plant to outlet } o \text{ in period } t$

$s_{t,p}: \text{number of plant } p \text{ stored in period } t$

$h_{p}: \text{cost of storing plant } p \text{ in period } t$

$X_{t,p}: \text{the max number of plant } p \text{ manufatured in period } t$

$Z_{t,o}: \text{the max number of plant } p \text{ sold to outlet } o$

$S: \text{the max number of plant stored}$

$\text{objective function:}$
$$
\min(\Sigma^{T}_{t=1}\Sigma^{O}_{o=1}r_{t,o}z_{t,o} - \Sigma^{T}_{t=1}\Sigma^{P}_{p=1}m_{t,p}x_{t,p} - \Sigma^{T}_{t=1}\Sigma^{P}_{p=1}\Sigma^{O}_{o=1}c_{p,o}y_{t,p,o} - \Sigma^{T}_{t=1}\Sigma^{P}_{p=1}h_{p}s_{t,p})
$$
$\text{constraints:}$
$$
x_{t,p} + s_{t-1,p} = \Sigma^{O}_{o=1}y_{t,p,o} + s_{t,p} \\
\Sigma^{P}_{p=1}y_{t,p,o} \ge z_{t,o} \\
x_{t,p} \le X_{t,p} \\
z_{t,o} \le Z_{t,o} \\
s_{t,p} \le S \\
x_{t,p}, y_{t,p,o}, z_{t,o}, s_{t,p} \ge 0
$$



##### Question 4

(a)

<img src=".\q4_a.png" alt="q4_a" style="zoom:33%;" />

(b)

$x_{i,j}: \text{number of the units from } i \text{ to } j$

$c_{i,j}: \text{cost of the flow from } i \text{ to } j \text{ per unit}$

$u_{i,j}: \text{max units of the flow from } i \text{ to } j$

$b_{i}: \text{supply for node } i$

$\text{objective function:}$
$$
\min(\Sigma_{i,j\in A}c_{i,j}x_{i,j})
$$
$\text{constraints:}$
$$
\Sigma_{k\in IN(i)}x_{k,i} + b_{i} = \Sigma_{j\in OUT(i)}x_{i,j}, \quad \forall i\in A \\
0 \le x_{i,j} \le u_{i,j}, \quad \forall i,j\in A \\
b_i=\begin{cases}
b_s, & i = s, \\
-b_s, & i = t, \\
0, & i \ne s, t.
\end{cases}
$$

(c)

codes:



report:





##### Question 5

$\text{objective function:}$
$$
\min(\Sigma^{23}_{t=0}c_{t}x_{t})
$$
$\text{constraints:}$
$$
x_{t} = \Sigma^{t+5}_{t}y_{t,(i\ mod\ 24)},\quad t = 0, 1, \cdots, 23 \\
\Sigma^{t}_{k=t-8}x_{k\ mod\ 24} - \Sigma^{t-3}_{k=t-5}y_{k\ mod\ 24,t} \ge r_t,\quad t = 0, 1, \cdots, 23 \\
x_{t} \ge 0,\quad t = 0, 1, \cdots, 23
$$





