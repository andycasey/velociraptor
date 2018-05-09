
data  {
  int N; // number of sources
  int M; // columns in the design matrix
  real rv_variance[N]; // radial velocity variance on single epoch measurements
  matrix[N, M] design_matrix; // design matrix 
}

transformed data {
  real max_rv_variance;
  max_rv_variance = max(rv_variance);
}

parameters {
  real<lower=0, upper=1> theta;
  vector<lower=0, upper=1e12>[M] mu_coefficients;
  vector<lower=0, upper=1e12>[M] sigma_coefficients;
}

transformed parameters {
  real<lower=0> rvf_mu[N];    // RV floor mu
  real<lower=0> rvf_sigma[N]; // RV floor scatter
  for (n in 1:N) {
    rvf_mu[n] = dot_product(design_matrix[n], mu_coefficients);
    rvf_sigma[n] = dot_product(design_matrix[n], sigma_coefficients);
  }
}

model {
  theta ~ beta(1, 5);
  for (n in 1:N) {
    target += log_mix(theta,
                      uniform_lpdf(rv_variance[n] | 0, max_rv_variance),
                      normal_lpdf(rv_variance[n] | rvf_mu[n], rvf_sigma[n]));
    }
}

generated quantities {
  real log_ps1;
  real log_ps2;
  real log_membership_probability[N];
  for (n in 1:N) {
    log_ps1 = log(theta) 
            + uniform_lpdf(rv_variance[n] | 0, max_rv_variance);
    log_ps2 = log1m(theta)
            + normal_lpdf(rv_variance[n] | rvf_mu[n], rvf_sigma[n]);
    log_membership_probability[n] = log_ps1 - log_sum_exp(log_ps1, log_ps2);
  }
}

