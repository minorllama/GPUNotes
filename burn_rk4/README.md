# Burn RK4 Harmonic Oscillator Solver

This Rust module uses the **Burn** deep learning framework (with an `NdArray` backend) and **PyO3** to solve the classical ordinary differential equation (ODE) for an ideal, undamped harmonic oscillator. It exposes the solver to Python. The code solves the **undamped, unforced Simple Harmonic Oscillator** equation:

$$\frac{d^2x}{dt^2} + x = 0$$

where $x$ represents the position of the oscillator. 

This is a barebones setup to get started with writing rust extensions for python. It's not meant to be comprehensive, and is more of a self note. 

### System Dynamics
To compute this numerically, the second-order differential equation is split into a system of two coupled first-order ODEs by introducing velocity $\left(v = \frac{dx}{dt}\right)$:

$$\frac{dx}{dt} = v$$
$$\frac{dv}{dt} = -x$$

In the vector field notation implemented within the `oscillator_system` function:

$$\frac{d\mathbf{y}}{dt} = \begin{bmatrix} v \\\\ -x \end{bmatrix}, \quad \text{where } \mathbf{y} = \begin{bmatrix} x \\\\ v \end{bmatrix}$$

### Analytical Solution & Setup
* **Physical Constants:** The system parameters are normalized such that the mass ($m = 1$) and the spring constant ($k = 1$), leading to a natural angular frequency $\omega_0 = 1$ and an exact period of $T = 2\pi$.
* **Initial Conditions:** Hardcoded in the initialization state as $x(0) = 1.0$ and $v(0) = 0.0$.
* **Expected Result:** The exact analytical solution for this system is:
  $$x(t) = \cos(t)$$
  $$v(t) = -\sin(t)$$

---

## Numerical Integration Method

The code integrates the system using the **Classical Runge-Kutta 4th Order (RK4)** method. For each time step $dt$, it evaluates the state derivative at four intermediate points to achieve $O(dt^4)$ local accuracy:

```rust
let k1 = oscillator_system(y.clone());
let k2 = oscillator_system(y.clone() + k1.clone() * (dt / 2.0));
let k3 = oscillator_system(y.clone() + k2.clone() * (dt / 2.0));
let k4 = oscillator_system(y.clone() + k3.clone() * dt);
y = y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
