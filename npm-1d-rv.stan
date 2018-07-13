data {
    int<lower=1> N; // number of data points
    int<lower=1> D; // number of dimensions
    real y[N, D]; // the data points.
}

parameters {
    real<lower=0, upper=1> theta; // the mixing parameter
    real<lower=0.50, upper=15> mu_single; // single star distribution mean
    real<lower=0.05, upper=10> sigma_single; // single star distribution sigma
    real<lower=0.20, upper=1.6> sigma_multiple; // multiplcity log-normal distribution sigma
    real<lower=log(mu_single + sigma_single) + pow(sigma_multiple, 2)> mu_multiple;
}

model {
    theta ~ beta(5, 5);
   
    for (n in 1:N) {
        target += log_mix(theta,
                          normal_lpdf(y[n] | mu_single, sigma_single),
                          lognormal_lpdf(y[n] | mu_multiple, sigma_multiple));
    }
}


