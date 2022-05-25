# Echo-state-networks-and-online-learning
This project includes two parts. In the first part of the project I implement a standard Echo State Network (ESN) model without using leaky-integrator neurons. I implement training of the read-out weights by means of the standard regularized least square method. I perform some simulations by considering a K step ahead forecasting task (evaluating several values of K, the forecasting on the "2sine" and "lorentz" time series. In order to train ESNs, I create output pairs starting from the time series.

In the second part of the project, I implement a version of the ESN model that is trained online by means of gradient descent (update weights after the presentation of each input data point). I perform simulations to compare the performance on K-step ahead forecasting tasks with respect to the standard ESN implementation(first part of the project) using the same time series. It is taken into account multiple K values for the forecasting horizon and process the "2sine" and "lorenz" time series. All hyperparameters ahan significantly effect the performance are fine tuned.
