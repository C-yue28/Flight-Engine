# Dynamics Module Documentation

## Overview

The dynamics module implements the Newton-Euler equations of motion for 6DOF dynamics (3 degrees of both translational/rotational freedom)

#### Notes

Most of the complex dynamics equations were directly sourced from external resources and AI tools. The documentation here is my own work but please do not fully trust whatever I have written here as I have essentially just written verbatim what I have read from websites and Google search summaries.

- **Coordinate system**: All calculations use the body frame NED conventions
- **SI Units**
- **Quaternion normalization**: Quaternions are automatically normalized to avoid drift

## Mathematical Data Structures

### State Vector

The complete state vector contains 13 elements:

- $p_n, p_e, p_d$: Position in inertial frame (North-east-down convention) (m)
- $u, v, w$: Velocity components in body frame (m/s)
- $q_0, q_1, q_2, q_3$: Attitude quaternion
- $p, q, r$: Angular velocity components - body frame (rad/s)

### Translational Dynamics

The translational dynamics equation in the body frame:

$$
\frac{dv}{dt} = \vec{a} - \omega \times v
$$

Where:
- $v = [u, v, w]^T$ is the body-frame velocity
- $\vec{a}=\frac{\vec{F}}{m}$ is the acceleration in body frame
- $\omega = [p, q, r]^T$ is the angular velocity (rad/s)
- The final term accounts for the fact that our reference frame is rotating with respect to the inertial

The total force is the sum of all external forces:

$$
\vec{F} = \vec{F}_{aero} + \vec{F}_{prop} + \vec{F}_{gravity}
$$

### Rotational Dynamics

The rotational dynamics equation (Euler's equation):

$$
\frac{d\omega}{dt} = I^{-1}(M - \omega \times \left(I\omega\right) - \frac{dI}{dt}\omega)
$$

Where:
- $I$ is the inertia tensor ($kg \cdot m^2$)
- $M$ is the total moment (Nm)
- $\omega \times I\omega$ is the gyroscopic moment
- $\frac{dI}{dt}\omega$ is the inertia change moment

The total moment is:

$$
M = M_{aero} + M_{prop} + M_{gyroscopic}
$$

### Kinematics

#### Position Kinematics

Position derivatives in the inertial frame:

$$
\frac{dp}{dt} = R(q)^T v
$$

The $R(q)$ is simply the rotation matrix to account for the change in reference frame from body to inertial

#### Quaternion Kinematics

Quaternion rate equation:

$$
\frac{dq}{dt} = \frac{1}{2} \Omega(\omega) q
$$

Where the (evil) quaternion rate matrix is:

$$
\Omega(\omega) = \begin{bmatrix}
0 & -p & -q & -r \\
p & 0 & r & -q \\
q & -r & 0 & p \\
r & q & -p & 0
\end{bmatrix}
$$

In (even more evil) component form:

$$
\begin{bmatrix}
\dot{q}_0 \\ \dot{q}_1 \\ \dot{q}_2 \\ \dot{q}_3
\end{bmatrix}
= \frac{1}{2}
\begin{bmatrix}
-q_1 p - q_2 q - q_3 r \\
q_0 p + q_2 r - q_3 q \\
q_0 q - q_1 r + q_3 p \\
q_0 r + q_1 q - q_2 p
\end{bmatrix}
$$

### Variable Mass Properties

Fuel consumption causes mass to decrease over time, also affecting the inertia moments.

$$
m(t) = m_0 - \int_0^t \dot{m}(\tau) d\tau
$$

$$
I(t) = I_0 - \int_0^t \dot{I}(\tau) d\tau
$$

## Numerical Integration

### Runge-Kutta 4th Order (RK4)

$$
\begin{aligned}
k_1 &= f(t_n, y_n) \\
k_2 &= f(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_1) \\
k_3 &= f(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_2) \\
k_4 &= f(t_n + h, y_n + hk_3) \\
y_{n+1} &= y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

- Accuracy: O(h⁴) (according to Wikipedia)

## API Reference

### EquationsOfMotion

**Init Parameters:**
- `mass`: Initial mass (kg)
- `inertia`: Initial 3×3 inertia tensor ($kg \cdot m^2$)

### set_aerodynamic_model()

Set the aerodynamic model for physics computation

### set_propulsion_system()

Set the propulsion system for thrust simulation

### set_gravity_model()

Set the gravity model - gravitational force computation.

### set_wind_model()

Set the wind model for atmospheric turbulence and wind shear.

### set_mass_properties()

**Parameters:**
- `mass`: Current mass (kg)
- `inertia`: Current inertia tensor ($kg \cdot m^2$)
- `mass_rate`: Mass rate of change (kg/s, positive for fuel burn)
- `inertia_rate`: Inertia rate of change ($kg \cdot m^2/s$)

Updates in systems where mass is nonconstant (i.e. fuel consumption)

### derivatives()

Compute all state derivatives(position, velocity, etc) and returns as vector

**Parameters:**
- `state`: Current state vector
- `**kwargs`: Additional parameters (control_deflections, density, mach, reynolds)

### integrate()

Integrate state forward by one time step.

**Parameters:**
- `state`: Current state vector
- `dt`: Time step (s)
- `**kwargs`: Additional parameters for derivatives