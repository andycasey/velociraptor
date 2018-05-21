
// Mixture model for the minimum detectable radial velocity variance from a
// single epoch Gaia/RVS measurement

data  {
  int N; // number of sources
  int M; // columns in the design matrix
  real rv_variance[N]; // radial velocity variance on single epoch measurements
  row_vector[M] design_matrix[N]; // design matrix for the rv jitter
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

model {
  theta ~ beta(1, 5);

  for (n in 1:N) {
    target += log_mix(
      theta, 
      background_uniform_lpdf,
      normal_lpdf(rv_variance[n] | 
        dot_product(design_matrix[n], mu_coefficients), 
        dot_product(design_matrix[n], sigma_coefficients)));
  }
}