# Numerov method
We perform a transform allowing for Numerov integration of the radial Schrodinger equation.

## Exponential Grid
$$
r = R_p (\exp (j \sigma) -1) +r_0
$$
where $R_p$ is a scaling parameter, $\sigma$ is the grid spacing in the exponential grid, $j$ is the grid index, and $r_0$ is an offset to avoid singularity at the origin.

$$
j \sigma  =x 
$$

### scaling parameter
The $R_p$ parameter dominates the grid from linear to exponential.

When $j \sigma \ll 1$,
$$
r \approx R_p (j \sigma) + r_0
$$
is a linear grid with spacing $R_p \sigma$.

When $j \sigma \gg 1$,
$$
r \approx R_p \exp (j \sigma) 
$$
is an exponential grid.

the derivative with respect to $x$ is
$$
\frac{\mathrm{d}r}{\mathrm{d}x} = R_p \exp (x) = f(x)\\
\frac{\mathrm{d}^2r}{\mathrm{d}x^2} = R_p \exp (x) = f(x)
$$

## Radial Schrodinger Equation
The radial Schrodinger equation is
$$
-\frac{1}{2} \frac{\mathrm{d}^2u}{\mathrm{d}r^2} + [V_{\text{eff}}(r) - E] u(r) = 0
$$
where $u(r) = \frac{R(r)}{r}$

Let 
$$
Q(r) = 2[V_{\text{eff}}(r) - E]
$$

### Chain rules
$$
\frac{\mathrm{d}u}{\mathrm{d}r} = \frac{\mathrm{d}u}{\mathrm{d}x} \frac{1}{f(x)} \\
\frac{\mathrm{d}^2u}{\mathrm{d}r^2} = \frac{1}{f^{2}(x)} \frac{\mathrm{d}^2u}{\mathrm{d}x^2} - \frac{f'(x)}{f^{3}(x)} \frac{\mathrm{d}u}{\mathrm{d}x}
$$

Thus
$$
\frac{\mathrm{d}^2u}{\mathrm{d}x^2} - \frac{f'(x)}{f(x)} \frac{\mathrm{d}u}{\mathrm{d}x} = f^{2}(x) Q(r) u(x) 
$$

## Eliminating the first derivative
let 
$$
v(x) = \frac{u(x)}{\sqrt{f(x)}}
$$

Then
$$
\frac{\mathrm{d}u}{\mathrm{d}x} = \frac{f'(x)}{2 \sqrt{f(x)}} v(x) + \sqrt{f(x)} \frac{\mathrm{d}v}{\mathrm{d}x}\\
\frac{\mathrm{d}^2u}{\mathrm{d}x^2} =  \cdots
$$

gives
$$
\frac{\mathrm{d}^2v}{\mathrm{d}x^2}= [f^{2}(x) Q(r) +\frac{1}{4}] v(x)
$$
