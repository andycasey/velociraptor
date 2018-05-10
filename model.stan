
// Mixture model for the minimum detectable radial velocity variance from a
// single epoch Gaia/RVS measurement

// Sampling cost: ~155 seconds for ~1e4 stars w/ 2 chains and 2000 iterations
//                ~33 hours for 7,224,631 stars w/ 2 chains and 2000 iterations

data  {
  int N; // number of sources
  int M; // columns in the design matrix
  real rv_variance[N]; // radial velocity variance on single epoch measurements
  row_vector[M] design_matrix[N];
}

transformed data {
  real background_uniform_lpdf;
  background_uniform_lpdf = uniform_lpdf(1 | 0, max(rv_variance));
}

parameters {
  real<lower=0, upper=1> theta;
  vector<lower=0>[M] mu_coefficients;
  vector<lower=0>[M] sigma_coefficients;
}

transformed parameters {
  real rvf_mu[N];    // RV floor mu
  real rvf_sigma[N]; // RV floor scatter
  for (n in 1:N) {
    rvf_mu[n] = dot_product(design_matrix[n], mu_coefficients);
    rvf_sigma[n] = dot_product(design_matrix[n], sigma_coefficients);
  }
}

model {
  theta ~ beta(1, 5);
  for (n in 1:N)
    target += log_mix(theta, 
                      background_uniform_lpdf,
                      normal_lpdf(rv_variance[n] | rvf_mu[n], rvf_sigma[n]));
}

generated quantities {
  real log_ps1;
  real log_ps2;
  real log_membership_probability[N];

  log_ps1 = log(theta) + background_uniform_lpdf;

  for (n in 1:N) {
    log_ps2 = log1m(theta)
            + normal_lpdf(rv_variance[n] | rvf_mu[n], rvf_sigma[n]);
    log_membership_probability[n] = log_ps1 - log_sum_exp(log_ps1, log_ps2);
  }
}