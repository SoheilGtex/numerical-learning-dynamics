# Phase 2 Mathematical Notes

## Least-squares objective

Given a design matrix $X\in\mathbb{R}^{m\times n}$ and observations $y\in\mathbb{R}^m$, least squares solves

$$
\min_{\beta\in\mathbb{R}^n}\|X\beta-y\|_2^2.
$$

The Phase 1 convention is preserved: public solver functions receive predictors without an intercept by default and prepend one column of ones. Passing `fit_intercept=False` means the caller has already supplied the complete design matrix.

## Normal equations

Let

$$f(\beta)=\|X\beta-y\|_2^2=(X\beta-y)^T(X\beta-y).$$

Differentiating gives

$$
\nabla f(\beta)=2X^T(X\beta-y).
$$

At a stationary point,

$$
X^TX\hat\beta=X^Ty.
$$

The implementation solves this system with `numpy.linalg.solve`; it never forms $(X^TX)^{-1}$. It rejects rank-deficient design matrices rather than silently replacing the method. Forming $X^TX$ can worsen the 2-norm condition number approximately by squaring it, which is one reason this method can be numerically sensitive. Phase 2 records a small conditioning diagnostic but does not perform the systematic study reserved for Phase 3.

## QR factorization

For a reduced QR factorization of a full-column-rank matrix,

$$X=QR,$$

where $Q$ has orthonormal columns and $R$ is square upper triangular. For reduced QR, the residual decomposes as

$$
\|X\beta-y\|_2^2
=
\|R\beta-Q^T y\|_2^2
+
\|(I-QQ^T)y\|_2^2.
$$

The orthogonal-complement term is independent of $\beta$. Therefore minimizing the full least-squares residual is equivalent to minimizing $\|R\beta-Q^T y\|_2^2$. For a full-column-rank system, the least-squares coefficients satisfy

$$
R\hat\beta=Q^Ty.
$$

The implementation uses `numpy.linalg.qr` with reduced, unpivoted QR and rejects detected rank deficiency or underdetermined systems.

## SVD and pseudoinverse

For

$$X=U\Sigma V^T,$$

the Moore-Penrose pseudoinverse is

$$X^+=V\Sigma^+U^T,$$

so a minimum-norm least-squares solution is

$$\hat\beta=X^+y.$$

The implementation computes `numpy.linalg.svd` directly and constructs the reciprocal singular-value factors explicitly. With `rcond=None`, singular values at or below

$$
\tau=\epsilon\max(m,n)\sigma_{\max}
$$

are discarded. With an explicit `rcond`, the threshold is $\tau=\texttt{rcond}\,\sigma_{\max}$. The retained count is the effective numerical rank.

When $X$ is rank deficient, multiple coefficient vectors can produce the same fitted values because directions in the null space do not change $X\beta$. The pseudoinverse selects the solution with minimum Euclidean norm among the least-squares solutions.

## Diagnostics and interpretation

The residual norm is

$$\|X\hat\beta-y\|_2.$$

It measures fit error in observation space. It does not by itself imply accurate coefficient recovery. In ill-conditioned or rank-deficient systems, substantially different coefficient vectors can produce similar predictions.

The condition diagnostic uses

$$\kappa_2(X)=\frac{\sigma_{\max}}{\sigma_{\min}}$$

for a full-rank matrix and reports infinity when the smallest singular value is zero. Standardizing predictor columns changes the coordinate representation, so standardized coefficient vectors are not compared directly with the original ground truth.

## Conditioning and perturbation

For a full-column-rank matrix, the 2-norm condition number is

$$\kappa_2(X)=\frac{\sigma_{\max}(X)}{\sigma_{\min}(X)}.$$

The singular values of $X^TX$ are $\sigma_i(X)^2$. Consequently, in exact arithmetic,

$$\kappa_2(X^TX)=\kappa_2(X)^2.$$

A small smallest singular value therefore produces a large condition number and increased sensitivity to data or design perturbations. The condition number describes the sensitivity of the mathematical problem. Algorithmic stability is a separate property: an ill-conditioned problem can challenge every solver, while an unstable algorithm can add further error. Normal Equations explicitly form $X^TX$ and can amplify conditioning; QR and SVD avoid that formation, but they do not remove the underlying ill-conditioning.

For a relative response perturbation $\|\delta y\|_2/\|y\|_2$, the condition number provides a first-order scale for possible relative solution sensitivity. Actual amplification depends on the perturbation direction and problem geometry; it is not an equality of the form “error = condition number times perturbation.”

The Phase 3 experiments distinguish model fit (residual norm and RMSE), parameter recovery (relative coefficient error), conditioning (condition number and singular spectrum), algorithm stability (differences among solvers), and identifiability (whether coefficients are uniquely determined). Standardization uses $z_j=(x_j-\mu_j)/s_j$. If $\beta$ is a raw-coordinate coefficient vector and $\gamma$ is its standardized-coordinate counterpart, then

$$
\gamma_j=\beta_js_j,
\qquad
\gamma_0=\beta_0+\sum_j\beta_j\mu_j,
$$

with the inverse mapping $\beta_j=\gamma_j/s_j$ and $\beta_0=\gamma_0-\sum_j\beta_j\mu_j$. This transformation is required before comparing coefficients across coordinate systems.

## Phase 3 experimental observations

The conditioning study uses the fixed construction $x_2=x_1+\delta z$ for nine values of $\delta$ from $1$ through $10^{-8}$. It evaluates one exact-model realization and ten deterministic noisy realizations at each level. The measured design condition number ranges from approximately $3.50$ to $2.38\times10^8$. Up to floating-point effects, the measured Gram condition number follows $\kappa_2(X)^2$; at the smallest level the Gram matrix is numerically treated as singular.

At $\delta=10^{-8}$ in the exact model, Normal Equations gives relative coefficient error $3.59\times10^{-1}$ while its in-sample fitted-value RMSE is $5.98\times10^{-9}$. QR, SVD, and `lstsq` retain fitted-value RMSE near $10^{-15}$ and coefficient errors near $10^{-9}$. In the noisy replicates, coefficient errors grow strongly with conditioning while median in-sample fitted-value RMSE remains near $0.143$ at the highest condition level. This separates fitted-value behavior from parameter recovery for these controlled synthetic systems. It does not test generalization performance; true out-of-sample forecasting belongs to a later phase.

For every nonzero $\delta$, the construction $x_2=x_1+\delta z$ satisfies

$$
\operatorname{span}\{x_1,x_2,x_3\}=\operatorname{span}\{x_1,z,x_3\}.
$$

Thus the mathematical column space is unchanged for nonzero $\delta$, although its coordinate representation becomes increasingly ill-conditioned as $\delta$ approaches zero. Least-squares fitted values are projections $\hat y=P_Xy$, so stable solvers can exhibit stable fitted-subspace behavior alongside unstable parameter coordinates. This is a mathematical statement about the construction; exact floating-point invariance is not assumed.

The response perturbation experiment uses fixed perturbation directions and magnitudes from $0$ through $10^{-2}$, scaled in the Euclidean vector 2-norm. The design-matrix perturbation uses the spectral matrix 2-norm explicitly:

$$
\frac{\|\Delta X\|_2}{\|X\|_2}=m,
\qquad
\Delta X=m\|X\|_2\frac{D}{\|D\|_2}.
$$

Its median amplification rises from approximately $0.334$ at $\delta=1$ to approximately $5.414\times10^3$ at $\delta=10^{-6}$ for the selected representative regimes. The response amplification for QR, SVD, and `lstsq` rises from approximately $0.226$ at $\delta=1$ to approximately $9.84\times10^6$ at $\delta=10^{-8}$. These values are directional measurements, not universal bounds.

# Phase 4 — Regularization and Uncertainty

## Centered Ridge/Tikhonov regularization

For predictors without an intercept, let $X_c=X-\bar x^T$ and $y_c=y-\bar y$. The zeroth-order Tikhonov problem is

$$\min_b \|X_cb-y_c\|_2^2+\lambda\|b\|_2^2,$$

where only slopes are penalized. The intercept is reconstructed as $\hat\beta_0=\bar y-\bar x^T\hat b$.

For $X_c=U\Sigma V^T$, the explicit SVD solution is

$$
\hat b_\lambda=V\operatorname{diag}\left(\frac{\sigma_i}{\sigma_i^2+\lambda}\right)U^Ty_c.
$$

Equivalently, the centered slope satisfies

$$
(X_c^TX_c+\lambda I)\hat b_\lambda=X_c^Ty_c.
$$

The coefficient inversion factor is $\sigma_i/(\sigma_i^2+\lambda)$. The hat-matrix shrinkage factor is $f_i(\lambda)=\sigma_i^2/(\sigma_i^2+\lambda)$. At $\lambda=0$ and nonzero $\sigma_i$, $f_i=1$. Increasing $\lambda$ suppresses all singular directions, with the smallest-singular-value directions suppressed most strongly relative to their unstable unregularized inverse factors. Ridge changes the estimation problem; it does not repair or remove the ill-conditioning of the original unregularized inverse problem.

## Effective degrees of freedom and conditioning

With an unpenalized intercept,

$$\operatorname{df}(\lambda)=1+\sum_i\frac{\sigma_i^2}{\sigma_i^2+\lambda}.$$

The leading one represents the intercept. For a full-rank centered predictor matrix,

$$\kappa_2(X_c^TX_c+\lambda I)=\frac{\sigma_{\max}^2+\lambda}{\sigma_{\min}^2+\lambda}.$$

It decreases as positive regularization is added for a fixed design, but this concerns the regularized system, not the original inverse problem. The dimension-aware grid uses $\lambda=\alpha\sigma_{\max}^2$, where $\sigma_{\max}$ is computed from the standardized training design. Standardization statistics are computed on training observations only, then applied unchanged to validation and test observations. Fitted coefficients are transformed back to raw coordinates before coefficient comparisons.

## Selection and bias-variance analysis

Validation selection minimizes validation RMSE using only the training and validation partitions. Generalized Cross-Validation uses only training data:

$$
\operatorname{GCV}(\lambda)=\frac{\operatorname{RSS}(\lambda)/n}{\left(1-\operatorname{df}(\lambda)/n\right)^2}.
$$

The oracle coefficient selector minimizes raw-coordinate coefficient error using the known synthetic $\beta^\star$. It is explicitly a synthetic diagnostic only and is not available as a practical selection method.

For fixed $X$ and zero-mean noise, Ridge is a linear estimator in $y$. Therefore its conditional expected coefficient vector can be obtained exactly, up to floating-point error, by fitting the same estimator to $E[y\mid X]=X\beta^\star$. The bias–variance experiment uses this noiseless fit for conditional relative bias and bias-squared, rather than treating the finite 30-replicate sample mean as bias. Empirical coefficient variance is computed around the exact conditional expected estimator, while coefficient MSE is computed against $\beta^\star$; their finite-replicate decomposition gap is retained rather than forced to zero. At $\lambda=0$, correctly specified full-rank OLS is conditionally unbiased, although its variance can be enormous in an ill-conditioned design.

For the bias–variance visualization, bias-squared, coefficient variance, and coefficient MSE are each divided by $\|\beta^\star\|_2^2$. Relative bias squared, normalized coefficient variance, and normalized coefficient MSE are therefore dimensionless and directly comparable; the exact conditional expected estimator remains the bias definition.

Selection is completed before the test partition is accessed. The regularization path therefore records training and validation quantities only; final test RMSE is computed once for each selected validation, GCV, or synthetic-only oracle model. The bias–variance experiment uses the same dimensionless alpha grid but computes its absolute lambdas from the singular-value scale of its own centered standardized training design, rather than reusing lambdas from another split.

Regularization introduces bias intentionally. Across repeated Gaussian-noise realizations, the experiments report relative bias, coefficient variance, coefficient MSE, and held-out RMSE. The useful question is not whether Ridge recovers $\beta^\star$ exactly, but how $\lambda$ trades bias against variance, perturbation sensitivity, and held-out error. No monotonic trade-off is assumed without checking the generated results.

## Classical OLS uncertainty

The analytical interval machinery applies to full-rank OLS under $\epsilon\sim N(0,\sigma^2I)$. For $n$ observations and $p$ fitted parameters,

$$\hat\sigma^2=\frac{\operatorname{RSS}}{n-p},\qquad n>p.$$

The implementation computes the OLS covariance using an SVD rather than explicitly forming $(X^TX)^{-1}$:

$$\widehat{\operatorname{Cov}}(\hat\beta)=\hat\sigma^2 V\operatorname{diag}(\sigma_i^{-2})V^T.$$

For a new design vector $x_0$ including the intercept, the estimated conditional mean is $\hat\mu_0=x_0^T\hat\beta$ with standard error

$$\operatorname{SE}_{\mathrm{mean}}=\sqrt{x_0^T\widehat{\operatorname{Cov}}(\hat\beta)x_0}.$$

A two-sided Student-$t$ confidence interval is

$$\hat\mu_0\pm t_{1-\alpha/2,n-p}\operatorname{SE}_{\mathrm{mean}}.$$

For a future noisy observation,

$$\operatorname{SE}_{\mathrm{pred}}=\sqrt{\hat\sigma^2+x_0^T\widehat{\operatorname{Cov}}(\hat\beta)x_0},$$

so the prediction interval is wider than the corresponding conditional-mean interval.

Ridge estimates are biased. Therefore the OLS covariance and Student-$t$ formulas above are not exact frequentist Ridge confidence intervals. In Phase 4, OLS receives analytical interval machinery, while Ridge uncertainty is represented empirically through the bias-variance and stability studies.

## Coverage interpretation

The coverage experiment generates data exactly under the stated Gaussian linear-model assumptions. It reports nominal 95% coverage against empirical coverage over 200 deterministic Monte Carlo replications at fixed design points. Sampling variation is expected; empirical coverage need not equal 0.95 exactly, and the result does not transfer automatically to real student data.

## Leverage-defined uncertainty points

For a fixed full-rank OLS design, the leverage of an existing design row $x_i$ is the corresponding diagonal of the hat matrix, computed here from a stable reduced QR factorization. Uncertainty in the estimated conditional mean generally depends on

$$x_0^T(X^TX)^{-1}x_0,$$

so higher-leverage design points typically have larger mean-response standard errors. This is a typical relationship, not a strict monotonicity claim for every possible pair of points. Prediction intervals also contain the irreducible future-noise term $\sigma^2$, so leverage differences may be less visually dramatic for prediction intervals than for conditional-mean confidence intervals.


# Phase 5 — Forecasting Horizons, Ablation, and Robustness

## Information checkpoints and genuine generalization

The Phase 5 feature sets are nested:

$$
F_1\subset F_2\subset F_3\subset F_4,
$$

where each $F_k$ contains only information available at that synthetic forecasting checkpoint. The model at checkpoint $k$ is

$$
y=X^{(k)}\beta^{(k)}+\epsilon^{(k)}.
$$

For early checkpoints, $\beta^{(k)}$ is interpreted as the best linear projection for that information set. It is not generally valid to compare its coefficients with a subvector of the later full-data generating coefficients because omitted later variables can change the population projection. Phase 5 therefore emphasizes untouched test MAE, RMSE, and $R^2$. A single deterministic 60/20/20 student partition is reused across checkpoints within each replicate, and all standardization uses training observations only.

The checkpoints represent information availability within a semester; they are not time-series or autoregressive models. The feature ranges are illustrative synthetic choices and do not claim empirical realism for Kharazmi students or any educational population.

## Predictive ablation

For feature or transparent feature group $j$, the ablation quantity is

$$
\Delta\mathrm{RMSE}_j=\mathrm{RMSE}_{\mathrm{without}\ j}-\mathrm{RMSE}_{\mathrm{full}}.
$$

This measures predictive contribution conditional on the other modeled features. It is not a causal effect, structural parameter, or proof of educational importance. Correlated predictors can make individual ablation changes small even when a group carries useful information.

## Huber robustness

Huber loss is

$$
\rho_\delta(r)=
\begin{cases}
\tfrac12r^2,&|r|\le\delta,\\
\delta(|r|-\tfrac12\delta),&|r|>\delta.
\end{cases}
$$

The implementation uses IRLS. At each iteration, residuals with $|r_i|\le\delta$ receive weight one, while larger residuals receive approximately $\delta/|r_i|$. The scale is estimated from MAD and $\delta=1.345\,\widehat{\mathrm{scale}}$; zero-scale residuals receive an explicit numerical guard. Iteration stops when

$$
\frac{\|\beta_{t+1}-\beta_t\|_2}{\max(1,\|\beta_t\|_2)}<\mathrm{tolerance},
$$

and convergence status and iteration count are retained. Quadratic loss near zero preserves efficiency for ordinary residuals, while linear tail growth limits the influence of large response residuals.

A response outlier is unusual $y$ conditional on $x$; a high-leverage point is unusual $x$. Huber loss primarily limits large residual influence and does not automatically solve arbitrary high-leverage predictor contamination.

## Phase 5 interpretation

The clean repeated study uses 20 deterministic dataset seeds and evaluates OLS, validation-selected Ridge, and Huber on the same held-out partitions. The contamination study corrupts training data only and evaluates every model on the same clean validation/test observations. The reported robustness quantity is

$$
\mathrm{RMSE\ degradation}=\mathrm{RMSE}_{\mathrm{contaminated\ training}}-\mathrm{RMSE}_{\mathrm{clean\ training}}.
$$

Classical OLS intervals are not automatically extended to Ridge, Huber, or misspecified early-checkpoint models. Phase 5 does not claim actual student performance prediction, causal feature effects, universal Ridge superiority, universal Huber superiority, or deployment readiness.


## Phase 5 contamination preprocessing correction

Feature contamination is applied to selected raw training predictors before preprocessing. For each selected training row and cyclic predictor column, the observed raw value is increased by a finite amount of $6.0$; validation and test raw predictors remain unchanged. Means and scales are then recomputed from the observed contaminated training predictors and applied to all three partitions. Response contamination changes training targets only and leaves predictor preprocessing unchanged. The two contamination modes remain separate, and robustness is measured on clean held-out observations.
