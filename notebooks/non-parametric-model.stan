
    data {
        int<lower=1> N; // number of data points
        real y[N]; // the data points.
    }

    parameters {
        real<lower=0,upper=1> theta; // the mixing parameter
        real<lower=0> mu_single; // single star distribution mean
        real<lower=0> sigma_single; // single star distribution sigma
        real<lower=0> sigma_multiple; // multiplcity log-normal distribution sigma
        real<lower=log(mu_single)+pow(sigma_multiple,2)> mu_multiple; // multiplicity log-normal distribution mean
    }
    
    model {
        theta ~ beta(5, 5);
        mu_single ~ normal(0, 1);
        sigma_single ~ normal(0, 1);
        
        for (n in 1:N)
            target += log_mix(theta,
                              normal_lpdf(y[n] | mu_single, sigma_single),
                              lognormal_lpdf(y[n] | mu_multiple, sigma_multiple));
  }
