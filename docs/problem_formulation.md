# Problem Formulation

## Origin and motivation

The project originated from a prediction problem posed by **Dr. Jamshid Saeidian during a Numerical Analysis class at Kharazmi University**: whether a student's final Numerical Analysis grade could be predicted from early-semester information. The extension documented here was developed independently. No authorized real student dataset is currently available, so Phase 1 uses controlled synthetic observations and makes no claims about actual students.

The research motivation is numerical rather than merely predictive. A transparent linear model provides a small setting in which the design matrix, least-squares solution, known generating parameters, and numerical error can all be inspected before later work on stability, inverse problems, regularization, and uncertainty.

## Variables and design matrix

For observation $i$, the predictors represent prior GPA, prerequisite mathematics performance, attendance rate, first-quiz performance, and assignment performance. The target $y_i$ represents a modeled final Numerical Analysis grade. The feature matrix $Z\in\mathbb{R}^{n\times 5}$ contains these five predictors without an intercept. The design matrix is

$$
X = \begin{bmatrix} 1 & Z_{11} & \cdots & Z_{15}\\
\vdots & \vdots & & \vdots\\
1 & Z_{n1} & \cdots & Z_{n5}
\end{bmatrix}.
$$

The intercept is therefore included exactly once. The parameter vector is $\beta=(\beta_0,\ldots,\beta_5)^T$, and $y\in\mathbb{R}^n$ contains the target observations.

## Linear model and synthetic assumptions

The data-generating model is

$$
y = X\beta^\star + \epsilon,
$$

where $\beta^\star$ is known ground truth and $\epsilon$ is independent zero-mean Gaussian noise with standard deviation `noise_std`. The implemented coefficient vector is

$$
\beta^\star=(1.00,\ 0.18,\ 0.16,\ 2.00,\ 0.12,\ 0.16)^T.
$$

The five features are sampled independently using a local NumPy generator. Prior GPA and prerequisite grade lie in $[8,20]$, attendance lies in $[0.6,1]$, and quiz and assignment performance lie in $[5,20]$. Targets are not clipped. This preserves the exact linear generating relationship, but it also means that noise can produce values outside a nominal grade scale. A future bounded-target model would need to document its clipping or transformation and its resulting nonlinearity.

The random generator is `numpy.random.default_rng(seed)`, so the same configuration and seed produce identical arrays without relying on global random state.

## Ordinary least squares

The baseline solves

$$
\hat{\beta}=\arg\min_{\beta}\|X\beta-y\|_2^2
$$

using `numpy.linalg.lstsq`. The implementation deliberately does not manually implement normal equations, QR, SVD, pseudoinverse analysis, conditioning experiments, or regularization in Phase 1.

Because the synthetic generator exposes $\beta^\star$, parameter recovery can be assessed directly rather than inferred only from predictive error.

## Evaluation metrics

For test targets $y$ and predictions $\hat y$, the baseline reports mean absolute error

$$MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|,$$

root mean squared error

$$RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2},$$

and

$$R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.$$

The implementation rejects constant targets for which the $R^2$ denominator is zero. Since ground truth is available, it also reports relative coefficient error

$$E_\beta=\frac{\|\hat\beta-\beta^\star\|_2}{\|\beta^\star\|_2}.$$

A zero ground-truth coefficient vector is rejected for this relative metric.

## Evaluation design and limitations

The reproducible baseline uses the first 75% of the deterministic sample as training data and the remaining 25% as test data. This split verifies data generation, fitting, prediction, metrics, and ground-truth recovery. It is not a final statistically rigorous validation protocol, and no educational or causal conclusion can be drawn from the synthetic experiment.

## Roadmap

Later phases may study numerical solver comparisons, conditioning, perturbation and stability, regularization, uncertainty quantification, robustness, feature ablation, temporal forecasting, and nonlinear dynamics. Those topics are intentionally outside Phase 1.
