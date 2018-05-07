
data  {
  int N; // number of sources
  real rv_variance[N];
  real flux[N];
}

transformed data {
  real inverse_flux_sq[N]; 
  for (n in 1:N)
    inverse_flux_sq[n] = pow(flux[n], -2);

}

parameters {
  real<lower=0, upper=1> outlier_fraction;
  real<lower=0> c0; // coefficient in rv_var ~ f(flux)
  real c1; // coefficient in rv_var ~ f(flux)

  real<lower=1e-3> s0; // coefficient for intrinsic scatter in RV floor
  real<lower=0> s1;    // coefficient for intrinsic scatter in RV floor
}

transformed parameters {
  // calculate a RV floor mu for each point
  real<lower=0> rvf_mu[N];    // intrinsic mu for RV floor
  real<lower=0> rvf_sigma[N]; // intrinsic scatter for RV floor 
  for (n in 1:N) {
    rvf_mu[n] = c0 + c1 * inverse_flux_sq[n];
    rvf_sigma[n] = s0 + s1 * inverse_flux_sq[n];
  }
}

model {
  outlier_fraction ~ beta(0.5, 0.5);
  for (n in 1:N) {
    target += log_mix(outlier_fraction,
                      uniform_lpdf(rv_variance[n] | 0, 20),
                      normal_lpdf(rv_variance[n] | rvf_mu[n], rvf_sigma[n]));
    }
}