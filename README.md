# Phillips / Breguet Overcoil Generator

A numerical engineering tool for designing a **Phillips-compliant Breguet overcoil** for a mechanical watch balance spring.

The project translates classical terminal-curve theory into a modern workflow:

**watchmaking theory → numerical optimisation → geometry → CAD**

The current desktop application extends the Phillips calculation with the later **Bovay second-order criterion**, allowing several Phillips-compliant terminal curves to be compared before the selected geometry is sent to Autodesk Fusion through MCP.

## Main functions

The generator uses practical hairspring parameters including:

- inner diameter
- spring thickness and height
- number of turns
- winding factor
- attachment angle
- terminal-curve ratio and angle
- lift geometry
- balance frequency

The lower spring is an Archimedean spiral. The terminal curve is numerically optimised to satisfy the classical Phillips conditions:

$$
m_x = \rho_0^2
$$

$$
m_y = 0
$$

Because these conditions do not define one unique geometry, the current application can generate several Phillips-compliant solutions and compare them with Bovay's second-order descriptor.

## Phillips + Bovay

For a valid Phillips curve, the first-order Bovay descriptor approaches

$$
\rho_1 = 0.
$$

The second-order descriptor

$$
\rho_2
$$

can then distinguish between different Phillips-compliant terminal curves.

With a relatively simple geometry, for example around a curve ratio of **0.7**, the alternatives can be almost superimposed. With stronger deformation, for example around **0.9**, the geometrical differences become more visible and the second-order comparison becomes more discriminating.

The optimized MCP export can select the tested Phillips-compliant curve with the lowest $\rho_2$.

## Hairspring stiffness and balance inertia

The desktop application also connects the geometry to the oscillator.

For a rectangular hairspring section,

$$
J = \frac{h t^3}{12}
$$

and the approximate rotational stiffness is

$$
C = \frac{EJ}{L_\mathrm{active}}.
$$

The final piton allowance is excluded from the active length.

For the desired balance frequency $f$,

$$
f = \frac{1}{2\pi}\sqrt{\frac{C}{I}}
$$

and therefore

$$
I = \frac{C}{(2\pi f)^2}.
$$

The application displays $C$ in Nmm/rad and the required balance inertia $I$ in mg·cm². The watchmaker can adjust the **spring height** until the calculated inertia corresponds to the actual balance.

## CAD and MCP

The program works with both:

1. the finished three-dimensional Breguet overcoil;
2. the unbent manufacturing geometry.

The high-resolution mathematical geometry is reduced to a practical number of Fusion spline fit points and checked against the exact reference geometry.

The current desktop application can send both the standard Phillips geometry and the Bovay-optimized geometry directly to **Autodesk Fusion through MCP**.

## Plot export

Engineering plots can be exported as **PNG and SVG**, including the comparison of multiple Phillips-compliant terminal curves. SVG is particularly useful for technical publications because the curves and text remain vector based.

## Repository contents

### Python source — Version 5

`phillips_breguet_curve_final.py`

This earlier version is published as a readable reference implementation of the mathematical and numerical construction. It generates CSV geometry data and engineering plots.

### Windows application

The current **PyQt5 desktop application** is distributed as a compiled Windows executable and contains the newer interactive workflow:

- GUI parameter input
- Phillips/Bovay comparison
- optimized terminal-curve selection
- hairspring stiffness and balance-inertia calculation
- PNG/SVG plot export
- direct Autodesk Fusion MCP export

> **Important:** the Version 5 Python source and the compiled desktop application are not the same software revision. Version 5 is provided primarily as a transparent reference implementation of the underlying calculation.

## Running the Version 5 source

Python 3.10 or newer is recommended.

Install the dependencies:

```bash
pip install numpy scipy matplotlib
```

Run:

```bash
python phillips_breguet_curve_final.py
```

Generated CSV files and plots are written next to the script.

## Typical Version 5 parameters

```text
Inner diameter       0.550 mm
Spring thickness     0.034 mm
Spring height        0.120 mm
Turns                13
Winding factor       4.0
Attachment angle     110°
Curve ratio          0.70
Beta                 240°
```

These are engineering inputs, not universal design values. They must be adapted to the movement, balance and hairspring being designed.

## Engineering scope

This project calculates and compares **hairspring geometry**. The Bovay second-order comparison should not be interpreted as a direct prediction of positional rate error in seconds per day.

A complete prediction of oscillator behaviour would require the subsequent dynamic deformation theory, including hairspring reactions and their effect on the period, followed by experimental validation.

## Background

The project grew from a study of historical construction methods for Breguet overcoils.

Phillips formulated mathematical conditions for terminal curves in the nineteenth century. Léopold Defossez later documented the theory for watchmaking education. Modern numerical optimisation makes it possible to solve these conditions directly, generate multiple valid solutions and compare them using Bovay's later formulation.

The result is not a replacement for classical watchmaking theory. It makes that theory **executable, testable and directly usable in modern CAD**.

## Author

**Kilian Eisenegger**

Watchmaking engineering, numerical methods and CAD automation.

## Disclaimer

This software is an engineering and educational tool. Calculated geometries and oscillator parameters should be validated for the intended movement, material, manufacturing process and operating conditions before use in production.


