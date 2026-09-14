# Assignment 6

[![Assignment 6 tests](https://github.com/PGE323M/assignment6/actions/workflows/main.yml/badge.svg)](https://github.com/PGE323M/assignment6/actions/workflows/main.yml)

## Learning objectives

- Read reservoir observations with pandas and operate on arrays with NumPy.
- Implement least squares by forming and solving the normal equations.
- Distinguish a free-intercept model from a through-origin model.
- Use reviewed repository instructions to preserve a required mathematical method, not just a correct-looking answer.

Complete the five methods of `KozenyCarmen` in [assignment6.py](assignment6.py). Preserve the class name (including its historical spelling), method names, and arguments. This Python module is the authoritative graded implementation; no notebook is required.

Supplementary lecture: https://youtu.be/v1vbLJmQ43E. Background: [least squares](https://en.wikipedia.org/wiki/Least_squares).

## Model and data

[poro_perm.csv](poro_perm.csv) contains columns `porosity` and `permeability`. Porosity is a dimensionless fraction; retain the permeability units of the supplied observations. Do not change the CSV or its row order.

The Kozeny-Carmen relationship is

$$
f(\phi)=\frac{\phi^3}{(1-\phi)^2}, \qquad \kappa=m f(\phi).
$$

Here permeability, not the proportionality constant, is proportional to the porosity transform. First allow a free intercept:

$$
\kappa=\kappa_0+m f(\phi).
$$

For N observations, form the design matrix and response:

$$
A=\begin{bmatrix}1 & f(\phi_1) \\ & f(\phi_2) \\ \vdots & \vdots \\ & f(\phi_N) \end{bmatrix},
\qquad b=\begin{bmatrix}\kappa_1 \\ \kappa_2 \\ \vdots \\ \kappa_N \end{bmatrix}.
$$

Minimizing the sum of squared residuals gives the normal equations

$$
(A^T A)x=A^T b, \qquad x=[\kappa_0,m]^T.
$$

**Required method:** form both sides with NumPy and solve this system using [numpy.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html). Do not explicitly invert the matrix. Do not substitute `lstsq`, `pinv`, `polyfit`, `curve_fit`, or another high-level fitting routine. This assignment intentionally teaches normal-equation assembly; the mathematical method is part of correctness. Use vectorized NumPy/pandas operations, not loops, comprehensions, row-wise `apply`, or conditional branches in these five methods. Matrix products may use `@`, [numpy.dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html), or equivalent NumPy matrix multiplication.

Inputs are finite, with valid porosities `0 <= phi < 1`. For `least_squares`, assume a two-dimensional, full-column-rank NumPy array A of shape `(N, p)`, N >= p, and a one-dimensional NumPy response b of shape `(N,)`. The normal matrix is `(p, p)` and the right-hand side is `(p,)`; return coefficients of shape `(p,)`. You need not add input-validation branches or handle singular systems.

## Required interface

- `__init__(self, filename)`: read the supplied filename using `pandas.read_csv` into `self.df`, then call `self.kc_model()`. Do not hard-code the course CSV path.
- `kc_model(self)`: add the column named exactly `KC model` using the vectorized transform above; preserve the original columns and return `None`.
- `least_squares(self, A, b)`: return the normal-equation solution for arbitrary full-column-rank A, not only two-column fitting matrices.
- `fit(self)`: construct the `(N, 2)` design matrix with ones first and `KC model` second, extract the `(N,)` permeability response, and call `self.least_squares(A, b)`. Return a length-two array in **(intercept, slope)** order, not slope first.
- `fit_through_zero(self)`: use the mirrored-data construction below and call `self.least_squares(A, b)`. Return **only the scalar slope**. Do not alter the stored observations.

## Through-origin fit by mirrored data

A nonzero fitted intercept may be undesirable for the physical proportionality model. For this exercise, append a negated copy of each predictor and its response. With `z = f(phi)` and measured permeability `kappa`, use paired rows `(z_i, kappa_i)` and `(-z_i, -kappa_i)`:

$$
A_{\rm mirror}=\begin{bmatrix} \mathbf{1} & z \\ \mathbf{1} & -z \end{bmatrix},
\qquad b_{\rm mirror}= \begin{bmatrix} \kappa \\ -\kappa \end{bmatrix}.
$$

The column of ones stays positive for **all 2N rows**. Do not negate the entire design matrix, remove the intercept column, or simply discard the free-fit intercept. The mirrored design has shape `(2N, 2)` and response `(2N,)`. The intercept should be zero to floating-point precision because predictor and response sums cancel. Return coefficient 1, the slope. Either ordering of the positive and negative row blocks is acceptable.

## Agent exercise: constrain the method

Assignment 5 extracted a repeated submission workflow into a skill. This starter now supplies that completed [submission skill](.github/skills/submit-assignment/SKILL.md) and a small baseline [AGENTS.md](AGENTS.md). Read and reuse them; do not recreate or edit the skill.

Before implementation, ask the agent:

> Read README.md, test.py, and poro_perm.csv. Propose a compact addition to AGENTS.md that preserves this assignment's required mathematical method. Explain the normal equations, array shapes, coefficient order, and the mirrored-data through-origin construction. Do not save files or implement code yet.

Review the proposed diff yourself. It should require the normal-equation solve, vectorized operations, reuse of `least_squares`, and correct intercept handling; it must preserve protected files, approval, and stop conditions. Explicitly approve or reject the addition before the agent saves it. An agent must never silently change its own governing instructions.

Start a fresh chat and ask which repository rules apply. Then request a bounded implementation plan that identifies the authoritative specification, target methods, data shapes, and numerical checks. Review and approve that plan before allowing edits to `assignment6.py` only. Inspect the resulting diff and explain why the mirrored intercept vanishes. Passing tests alone does not establish that you understand or reviewed the method.

Do not add a second skill or duplicate the submission procedure in `AGENTS.md`. Standing mathematical constraints belong in the small repository instruction file; the repeated submission procedure remains in the provided skill.

## Testing and evidence

Run the transparent public checks from the repository root:

```bash
python -m unittest -v
```

Tests cover the provided instruction/skill baseline, API and method constraints, normal-equation assembly, synthetic fits, and mirrored-data construction. Baseline artifact checks pass before your instruction addition; implementation checks initially fail on explicit stubs. Contract checks establish only a minimal syntax contract, not proof of student review or instruction quality. No private chat transcript or LLM prose grading is required.

Independently check `(intercept, slope)` order, residuals, array shapes, and the near-zero mirrored intercept. For a fitted least-squares solution, the residual `b - A @ x` should be orthogonal to the columns of A up to numerical roundoff. Interpret results in the original permeability units.

## Submission

Deliverables are exactly `AGENTS.md` and `assignment6.py`. Do not change `README.md`, `test.py`, `poro_perm.csv`, `environment.yml`, `.gitignore`, `.github/` (including the provided skill), or `.devcontainer/`.

When your reviewed instruction addition and implementation are complete, invoke:

> submit assignment 6

The provided skill checks changes, runs tests, stages only the two deliverables, commits, pushes, and verifies GitHub Actions. Independently confirm the pushed commit and Actions result.
