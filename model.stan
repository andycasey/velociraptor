
data  {
  int N; // number of sources
  int M; // columns in the design matrix for the rv floor mu
  int S; // columns in the design matrix for the rv floor scatter
  real rv_variance[N]; // radial velocity variance on single epoch measurements
  matrix[N, M] mu_design_matrix; // design matrix for rv floor mu
  matrix[N, S] sigma_design_matrix; // design matrix for rv floor sigma
}

transformed data {
  real max_rv_variance;
  max_rv_variance = max(rv_variance);
}

parameters {
  real<lower=0, upper=1> outlier_fraction;
  vector<lower=1e-3, upper=1e12>[M] mu_coefficients;
  vector<lower=1e-3, upper=1e12>[S] sigma_coefficients;
}

transformed parameters {
  real<lower=0> rvf_mu[N];    // RV floor mu
  real<lower=0> rvf_sigma[N]; // RV floor scatter
  for (n in 1:N) {
    rvf_mu[n] = dot_product(mu_design_matrix[n], mu_coefficients);
    rvf_sigma[n] = dot_product(sigma_design_matrix[n], sigma_coefficients);
  }
}

model {
  outlier_fraction ~ beta(0.5, 0.5);
  for (n in 1:N) {
    target += log_mix(outlier_fraction,
                      uniform_lpdf(rv_variance[n] | rvf_mu[n], max_rv_variance),
                      normal_lpdf(rv_variance[n] | rvf_mu[n], rvf_sigma[n]));
    }
}