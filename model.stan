
data  {
  int N; // number of sources
  int M; // columns in the design matrix for the mu floor
  int S; // columns in the design matrix for the scatter floor
  real rv_variance[N];
  real flux[N];
  matrix[N, M] mu_design_matrix;
  matrix[N, S] sigma_design_matrix;
}

// N, K,  and K
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
  // calculate a RV floor mu for each point
  real<lower=0> rvf_mu[N];    // intrinsic mu for RV floor
  real<lower=0> rvf_sigma[N]; // intrinsic scatter for RV floor 
  for (n in 1:N) {
    rvf_mu[n] = dot_product(mu_design_matrix[n], mu_coefficients);
    rvf_sigma[n] = dot_product(sigma_design_matrix[n], sigma_coefficients);
  }
}

model {
  outlier_fraction ~ beta(0.5, 0.5);

  for (n in 1:N) {
    target += log_mix(outlier_fraction,
                      uniform_lpdf(rv_variance[n] | 0, 400),
                      normal_lpdf(rv_variance[n] | rvf_mu[n], rvf_sigma[n]));
    }
}