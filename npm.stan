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

    for (d in 1:D) {
        real var_multiple;
        real bound_lower;
        real bound_upper;
        var_multiple = pow(sigma_multiple[d], 2);
        bound_lower = log(mu_single[d]);
        bound_upper = log(mu_single[d] + 5 * sigma_single[d]);
        mu_multiple[d] = bound_lower + var_multiple
                       + mu_multiple_uv[d] * (bound_upper - bound_lower);
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
