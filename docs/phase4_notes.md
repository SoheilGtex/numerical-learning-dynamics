# Phase 4 — Regularization and Uncertainty

## Centered Ridge/Tikhonov regularization

For predictors without an intercept, let

$$X_c=X-\bar x^T,\qquad y_c=y-\bar y.$$

The zeroth-order Tikhonov problem used here is

$$
\min_b \|X_cb-y_c\|_2^2+\lambda\|b\|_2^2,
$$

where only slopes are penalized. The intercept is reconstructed as

$$
\hat\beta_0=\bar y-\bar x^T\hat b.
$$

For $X_c=U\Sigma V^T$, the explicit SVD solution is

$$
\hat b_\lambda
=V\mathrm{diag}\left(\frac{\sigma_i}{\sigma_i^2+\lambda}\right)U^Ty_c.
$$

Equivalently, the centered slope satisfies

$$
(X_c^TX_c+\lambda I)\hat b_\lambda=X_c^Ty_c.
$$

The coefficient inversion factor is $\sigma_i/(\sigma_i^2+\lambda)$. The hat-matrix shrinkage factor is

$$
f_i(\lambda)=\frac{\sigma_i^2}{\sigma_i^2+\lambda}.
$$

At $\lambda=0$ and nonzero $\sigma_i$, $f_i=1$. Increasing $\lambda$ suppresses all singular directions, with the smallest-singular-value directions suppressed most strongly relative to their unstable unregularized inverse factors. Ridge changes the estimation problem; it does not repair or remove the ill-conditioning of the original unregularized inverse problem.

## Effective degrees of freedom and conditioning

With an unpenalized intercept, the effective degrees of freedom convention used in the experiments is

$$
\mathrm{df}(\lambda)=1+\sum_i\frac{\sigma_i^2}{\sigma_i^2+\lambda}.
$$

The leading one represents the intercept. For a full-rank centered predictor matrix, the regularized normal-system condition number is

$$
\kappa_2(X_c^TX_c+\lambda I)
=\frac{\sigma_{\max}^2+\lambda}{\sigma_{\min}^2+\lambda}.
$$

It decreases as positive regularization is added for a fixed design, but this concerns the regularized system, not the original inverse problem.

The dimension-aware grid uses $\lambda=\alpha\sigma_{\max}^2$, where $\sigma_{\max}$ is computed from the standardized training design. Standardization statistics are computed on training observations only, then applied unchanged to validation and test observations. Fitted coefficients are transformed back to raw coordinates before coefficient comparisons.

## Selection and bias-variance analysis

Validation selection minimizes validation RMSE using only the training and validation partitions. The selected model is then evaluated once on the untouched test partition. Generalized Cross-Validation uses only training data:

$$
\mathrm{GCV}(\lambda)=
\frac{\mathrm{RSS}(\lambda)/n}
{\left(1-\mathrm{df}(\lambda)/n\right)^2}.
$$

The oracle coefficient selector minimizes raw-coordinate coefficient error using the known synthetic $\beta^\star$. It is explicitly a synthetic diagnostic only and is not available as a practical selection method.

For fixed $X$ and zero-mean noise, Ridge is linear in $y$. Thus its exact conditional expected coefficient vector is obtained by applying the same estimator to $E[y\mid X]=X\beta^\star$. The reported relative bias and bias-squared use this noiseless expected-response fit, not the unstable mean of 30 noisy replicates. Empirical coefficient variance is measured around that expected estimator; coefficient MSE remains measured against $\beta^\star$. Under the model, $\mathrm{MSE}=\mathrm{variance}+\mathrm{bias}^2$ in expectation, with a finite-replicate decomposition gap allowed.

At $\lambda=0$, full-rank OLS is conditionally unbiased under the correctly specified classical linear model, although its coefficient variance can be enormous in an ill-conditioned design. Positive Ridge regularization introduces conditional bias intentionally. The useful question is how $\lambda$ trades this bias against variance, perturbation sensitivity, and held-out error. No monotonic trade-off is assumed without checking the generated results.

For visualization, the experiment divides bias-squared, coefficient variance, and coefficient MSE by $\|\beta^\star\|_2^2$. The resulting relative bias squared, normalized variance, and normalized MSE are dimensionless and directly comparable in the bias–variance figure.

## Classical OLS uncertainty

The analytical interval machinery applies to full-rank OLS under

$$
\epsilon\sim N(0,\sigma^2I).
$$

For $n$ observations and $p$ fitted parameters, the unbiased noise-variance estimator is

$$
\hat\sigma^2=\frac{\mathrm{RSS}}{n-p},\qquad n>p.
$$

The implementation computes the OLS covariance using an SVD rather than explicitly forming $(X^TX)^{-1}$:

$$
\widehat{\mathrm{Cov}}(\hat\beta)
=\hat\sigma^2 V\mathrm{diag}(\sigma_i^{-2})V^T.
$$

For a new design vector $x_0$ including the intercept, the estimated conditional mean is $\hat\mu_0=x_0^T\hat\beta$ with standard error

$$
\mathrm{SE}_{\mathrm{mean}}
=\sqrt{x_0^T\widehat{\mathrm{Cov}}(\hat\beta)x_0}.
$$

A two-sided Student-$t$ confidence interval is

$$
\hat\mu_0\pm t_{1-\alpha/2,n-p}\mathrm{SE}_{\mathrm{mean}}.
$$

For a future noisy observation, the prediction standard error is

$$
\mathrm{SE}_{\mathrm{pred}}
=\sqrt{\hat\sigma^2+x_0^T\widehat{\mathrm{Cov}}(\hat\beta)x_0},
$$

so the prediction interval is wider than the corresponding conditional-mean interval.

Ridge estimates are biased. Therefore the OLS covariance and Student-$t$ formulas above are not exact frequentist Ridge confidence intervals. In Phase 4, OLS receives analytical interval machinery, while Ridge uncertainty is represented empirically through the bias-variance and stability studies.

## Coverage interpretation

The coverage experiment generates data exactly under the stated Gaussian linear-model assumptions. It reports nominal 95% coverage against empirical coverage over 200 deterministic Monte Carlo replications at fixed design points. Sampling variation is expected; empirical coverage need not equal 0.95 exactly, and the result does not transfer automatically to real student data.
