data {
    int<lower=1> N; // number of data points
    int<lower=1> D; // number of dimensions
    real y[N, D]; // the data points.
}

parameters {
    real<lower=0,upper=1> theta; // the mixing parameter
    real<lower=0> mu_single[D]; // single star distribution mean
    real<lower=0> sigma_single[D]; // single star distribution sigma
    real<lower=0> sigma_multiple[D]; // multiplcity log-normal distribution sigma
    real<lower=0, upper=1> mu_multiple_uv[D]; // unit vector of the multiplicity log-normal distribution mean
}

transformed parameters {
    real mu_multiple[D];

    # Bound the log-normal such that at most 10% of the support is at the mean
    # of the normal.
    {
        real min_support;
        real min_mode;

        for (d in 1:D) {
            real bound_lower;
            real bound_upper;
            min_support = log(mu_single[d]) + 1.263404 * sigma_multiple[d];
            min_mode = log(mu_single[d] + 1 * sigma_single[d]) + pow(sigma_multiple[d], 2);

            bound_lower = fmax(min_mode, min_support);
            bound_upper = log(mu_single[d] + 5 * sigma_single[d]) + pow(sigma_multiple[d], 2);
            mu_multiple[d] = bound_lower + mu_multiple_uv[d] * (bound_upper - bound_lower);
        }
    }
}

model {

    theta ~ beta(5, 5);
    
    for (n in 1:N) {
        for (d in 1:D) {
            target += log_mix(theta,
                              normal_lpdf(y[n, d] | mu_single[d], sigma_single[d]),
                              lognormal_lpdf(y[n, d] | mu_multiple[d], sigma_multiple[d]));
        }
    }
}
