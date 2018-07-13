data {
    int<lower=1> N; // number of data points
    int<lower=1> D; // number of dimensions
    real y[N, D]; // the data points.
    real theta_bounds[2];
    real mu_single_bounds[2];
    real sigma_single_bounds[2];
    real sigma_multiple_bounds[2];
}

parameters {
    real<lower=theta_bounds[1],upper=theta_bounds[2]> theta; // the mixing parameter
    real<lower=mu_single_bounds[1], upper=mu_single_bounds[2]> mu_single; // single star distribution mean
    real<lower=sigma_single_bounds[1], upper=sigma_single_bounds[2]> sigma_single; // single star distribution sigma
    real<lower=sigma_multiple_bounds[1], upper=sigma_multiple_bounds[2]> sigma_multiple; // multiplcity log-normal distribution sigma
    real<lower=log(mu_single + sigma_single) + pow(sigma_multiple, 2)> mu_multiple;
    //    real<lower=0, upper=1> mu_multiple_uv[D]; // unit vector of the multiplicity log-normal distribution mean
}
/*
transformed parameters {
    real mu_multiple[D];

    // Bound the log-normal such that at most 10% of the support is at the mean
    // of the normal.
    {
     
        for (d in 1:D) {
            real bound_lower;
            real bound_upper;
            bound_lower = log(mu_single[d] + 1 * sigma_single[d]) + pow(sigma_multiple[d], 2);
            bound_upper = max_log_y;
            mu_multiple[d] = bound_lower + mu_multiple_uv[d] * (bound_upper - bound_lower);
        }
    }
}
*/

model {

    theta ~ beta(5, 5);
    
    for (n in 1:N) {
        target += log_mix(theta,
                          normal_lpdf(y[n] | mu_single, sigma_single),
                          lognormal_lpdf(y[n] | mu_multiple, sigma_multiple));
    }
}


