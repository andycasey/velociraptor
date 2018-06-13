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
}

transformed parameters {
    real mu_multiple[D]; // multiplicity log-normal distribution mean [fixed]

    for (d in 1:D)
        mu_multiple[d] = log(mu_single[d] + 3 * sigma_single[d]) + pow(sigma_multiple[d], 2);
}


model {

    theta ~ beta(5, 5);
    
    // Place a prior such that the mode of the log normal distribution is larger
    // than the median of the normal distribution
    /*
    for (d in 1:D)
        mu_multiple[d] ~ uniform(
            log(mu_single[d]) + pow(sigma_multiple[d], 2),
            log(mu_single[d] + 3 * sigma_single[d]) + pow(sigma_multiple[d], 2));

    */
    for (n in 1:N)
        for (d in 1:D)
            target += log_mix(theta,
                              normal_lpdf(y[n, d] | mu_single[d], sigma_single[d]),
                              lognormal_lpdf(y[n, d] | mu_multiple[d], sigma_multiple[d]));
}
