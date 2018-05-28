
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import velociraptor


sources = velociraptor.load_gaia_sources("data/rv-all.fits")
model, data_dict, init_dict, idx = velociraptor.prepare_model(S=1e4, **sources)

N, M = (data_dict["N"], data_dict["M"])
init_dict.update(
    mu_coefficients=[2.1465e-05, 1.4585e+02, 2.0827e+11, 7.7332e-08, 5.8626e+00],
    sigma_coefficients=[7.4405e-04, 3.9762e-01, 1.7293e+11, 4.1103e-04, 5.9489e+00]
)

theta = tf.clip_by_value(tf.Variable(0.5, dtype=tf.float64), 0, 1)
mu_coefficients = tf.clip_by_value(
    tf.Variable(np.atleast_2d(init_dict["mu_coefficients"]), dtype=tf.float64),
    0, np.inf)
sigma_coefficients = tf.clip_by_value(
    tf.Variable(np.atleast_2d(init_dict["sigma_coefficients"]), dtype=tf.float64),
    0, np.inf)

mu = tf.matmul(mu_coefficients, data_dict["design_matrix"].T)
ivar = tf.matmul(sigma_coefficients, data_dict["design_matrix"].T)**-2

log_p_s = tf.log(1.0 - theta) \
        - 0.5 * np.log(2 * np.pi) + 0.5 * tf.log(ivar) \
        - 0.5 * (data_dict["rv_variance"] - mu)**2 * ivar

log_p_b = tf.log(theta) \
        - np.log(np.max(data_dict["rv_variance"])) * np.ones((1, data_dict["N"]))

log_prob = tf.reduce_sum(tf.reduce_logsumexp(tf.concat([log_p_s, log_p_b], 0), 0))

tolerance = 1e-6
learning_rate = 1e-3
max_iterations = 100000

training_step = tf.train.AdamOptimizer(learning_rate).minimize(-log_prob)

cost_history = np.empty(shape=[1],dtype=float)

with tf.Session() as session:
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    print("Initial log prob: {0}".format(session.run(log_prob)))
    print("Initial gradient: {0}".format(
        session.run(tf.gradients(log_prob, 
                                [theta, mu_coefficients, sigma_coefficients]))))

    for iteration in range(max_iterations):
        session.run(training_step)
        cost_history = np.append(cost_history, session.run(log_prob))
        
        if iteration % 1000 < 1:
            print(iteration, cost_history[-1], session.run(theta))
        
        if np.abs(np.diff(cost_history[-2:])) < tolerance:
            break

    theta_value = session.run(theta)
    mu_coefficients_value = session.run(mu_coefficients)
    sigma_coefficients_value = session.run(sigma_coefficients)

