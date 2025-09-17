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







