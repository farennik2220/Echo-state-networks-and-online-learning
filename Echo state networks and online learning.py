#!/usr/bin/env python
# coding: utf-8

# # Echo state networks and online learning

# ## Model answer for the project

# *<font color='red'>Please note that this is just one of the many possible ways to correctly implement ESNs with batch and online training. This notebook does not contain the implementation of FORCE learning.</font>*.
# 

# Before starting, populate the namespace with pylab and include IPython functionality to better present certain output:

# In[4]:


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np

from IPython.display import clear_output


# ### Standard echo state network (ESN) model

# Read the '2sin' file from the archive to use in the first section. Convert it into a $N\times1$ matrix as it only contains a single feature:

# In[5]:


X_2sin = np.genfromtxt("2sin.txt", delimiter=",")

X_2sin = np.array([np.array((x, )) for x in X_2sin])

plt.title("The dataset 2sin")
plt.plot(X_2sin[1:], "b")
plt.show()


# Split it into training and testing sets, this is to ensure when it comes to evaluating the algorithm it is not overfitting data. I have chosen to split it into two equal parts:

# In[6]:


Xtr_2sin, Xte_2sin = np.split(X_2sin, 2)


# An Echo State Network (ESN) is a type of Neural network (NN) that build upon the concept of Reservoir Computing. This type of model contains many hidden nodes that are interconnected, and store an impression of previous states. This memory causes the 'echo', and allows it to be applied to time-series forecasting tasks.
# 
# The Model is described as:
# 
# \begin{align*}
# X_t = \phi(W^rx_{t-1}+W^iu_t+\epsilon)
# \end{align*}
# 
# Where:
#  - $X_t$ is the outputs from each neuron in the hidden layer at time $t$
#  - $W^r$ are the weights of the connections between neurons in the hidden layer
#  - $W^i$ are the weights from the input neurons into the hidden layer
#  - $u_t$ are the inputs at time $t$
#  - $\epsilon$ is a vector of additive white Gaussian noise with a small standard deviation
# 
# The prediction from the neural network is given by:
# 
# \begin{align*}
# z_t=W^ox_t
# \end{align*}
# 
# Where:
#  - $z_t$ is the prediction
#  - $W^o$ are the weights of each connection from the hidden layer
#  - $x_t$ is the outputs from each neuron in the hidden layer at time $t$
# 
# To train the network you have to go through the training dataset and calculate $X_t$ for each $t$, initializing $X_{(-1)}$ as a vector of zeros. This set of values forms $X^T$.
# 
# The weights $W^o$ are calculated from this by solving a standard regularized least-square problem:
# 
# \begin{align*}
# W^o=(X^TX)^{-1}X^Tt
# \end{align*}
# 
# To initialize the network first define a spectral radial function to generate $W^r$, where if $\lambda_1,...,\lambda_n$ are the eigenvalues of a matrix $A\in \mathbb{R}^{nxn}$
# 
# The spectral radius can be defined as:
# 
# \begin{align*}
# \rho(A)=max\{|\lambda_1|,...,|\lambda_n|\}
# \end{align*}

# In[7]:


def spectral_radius(A):
    '''
    compute the spectral radius of matrix A
    '''
    w, v = numpy.linalg.eig(A)

    return np.max(np.absolute(w))


# When implementing an ESN, other hyper-parameters need to be considered:
# 
# #### Reservoir size
# 
# From reading papers on the subject, it is general wisdom that a larger reservoir can obtain a better performance. This relationship is accepted as long as regularization methods are applied to prevent overfitting.
# 
# At the extreme case, a reservoir can become too large if the problem being solved is trivial, or when there is not enough available data:
# 
# \begin{align*}
# T < 1 + N_u + N_x
# \end{align*}
# 
# Because of this, I have decided to set the default size to be $\frac{1}{3}^{rd}$ the number of data points in the training data.

# In[8]:


def esn(Xtr, Xte, k, Nr=None, progress=True):
    # Number of inputs
    Ni = 1

    # Number of hidden neurons
    if Nr is None:
        Nr = int(Xtr.shape[0] / 3)
    
    # Weights of inputs to neurons
    Wi = np.random.uniform(-1,1,(Nr, Ni))
    
    # Weights between neurons in the hidden layer
    Wr = np.random.uniform(-1, 1, (Nr, Nr))

    # define the hyper-parameter `a`
    a = 1
    
    # set new weights between neurons in hidden layer
    Wr = a * Wr /  spectral_radius(Wr)
    
    X = [np.zeros(Nr)]
    
    # Iterate all steps and generate Xt for each
    for i in range(Xtr.shape[0] - k):
        # Optionally print progress for larger tasks
        if progress and i % 50 == 0:
            clear_output(wait=True)
            print("{:<8}{:>6.0%}".format("Training", i/Xtr.shape[0]))

        term1 = Wr @ X[-1]
        term2 = Wi @ Xtr[i]
        
        # Define the noise to be added to the network
        epsilon = np.random.normal(0, 0.1, size=Nr)
    
        X.append(np.tanh(term1 + term2 + epsilon))
    
    # Remove the initial t-1 zeros entry
    X = np.array(X[1:])
    
    # Lambda is a hyper-parameter representing the learning rate
    lbda = 0.0001
    
    # Calculate the output weights using linear regression
    Wo = (X.T @ X) + ((lbda**2) * np.identity(Nr))
    Wo = np.linalg.inv(Wo)
    Wo = (Wo @ X.T) @ Xtr[k:]
    Wo = Wo.T
    
    # Initiate testing
    x = np.zeros(Nr)
    Z = []
    
    # Go through testing and predict values
    for i in range(Xte.shape[0] - k):
        # Optionally print progress for larger tasks
        if progress and i % 50 == 0:
            clear_output(wait=True)
            print("{:<8}{:>6.0%}".format("Testing", i/Xte.shape[0]))

        term1 = Wr @ x
        term2 = Wi @ Xte[i]
        
        # Define the noise to be added to the network
        epsilon = np.random.normal(0, 0.1, size=Nr)
        
        x = np.tanh(term1 + term2 + epsilon)

        Z.append(Wo@x)
    
    if progress:
        clear_output()

    return np.array(Z)


# As an example of the alogirthm working, apply it to the '2sin' dataset:

# In[9]:


k = 1
Z = esn(Xtr_2sin, Xte_2sin, k)

plt.title("Echo state network predictions after trianing")
plt.axis((-10, Xte_2sin[k:].shape[0] + 10, -3, 3))
plt.plot(Xte_2sin[k:], "b", label="testing")
plt.plot(Z[:,0], "r.", label="predictions")
legend(loc=1)
plt.show()


# Changing the value of $k$ has a huge impact on the success of the ESN, as shown below. This can be attributed to the noise $\epsilon$ added:

# In[10]:


figure(figsize=(12,13))

k = 1
Z = esn(Xtr_2sin, Xte_2sin, k, progress=False)

plt.subplot(221)
plt.title("ESN model with k=1 step-ahead forecasting")
plt.axis((-10, Xte_2sin[k:].shape[0] + 10, -3, 3))
plt.plot(Xte_2sin[k:], "b", label="testing")
plt.plot(Z[:,0], "r.", label="predictions")
legend(loc=1)

k = 7
Z = esn(Xtr_2sin, Xte_2sin, k, progress=False)

plt.subplot(222)
plt.title("ESN model with k=5 step-ahead forecasting")
plt.axis((-10, Xte_2sin[k:].shape[0] + 10, -3, 3))
plt.plot(Xte_2sin[k:], "b", label="testing")
plt.plot(Z[:,0], "r.", label="predictions")
legend(loc=1)

plt.show()


# It would be useful to visually see the error between targets and predictions numerically.
# 
# Define a function to calculate normalized root mean squared. It is important to normalize the error when comparing two different datasets. To do this, we can first calculate the root-mean-square error: 
# 
# \begin{align*}
# MSE = \frac{\Sigma_{n=1}^N(t_{n} - p_{n})^2}{n}
# \end{align*}
# 
# And then square-rooting and dividing by the variance:
# 
# \begin{align*}
# NRMSE = \frac{\sqrt{MSE}}{var(t)}
# \end{align*}

# In[11]:


def nrmse(targets, predictions):
    mse = np.sqrt(((predictions - targets) ** 2).mean())
    return mse / np.var(targets)


# Cycle through a range of possible $k$ values and plot them:

# In[12]:


E = []
K = range(1, 20)

print("NRMSE for a range of k values:")
for k in K:
    Z = esn(Xtr_2sin, Xte_2sin, k, progress=False)
    E.append(nrmse(Xte_2sin[k:][5:], Z[5:]))
    print("k={:<6} e={:<6.5}".format(k, E[-1]))

plt.title("NRMSE for a range of $k$ values:")
plt.plot(K, E, "r")
plt.ylabel("NRMSE")
plt.xlabel("k-step ahead")
plt.xticks(np.arange(min(K), max(K)+1, 1.0))
plt.show()


# The output generated from the above cell changes every run, but in general it is observed that a $k$ value of 1 is optimal, and there is a slight dip around the 12-15 range. This is possibly due to the way the data oscilates, with it being able to use the predictable repetition as an indication of the next time step.

# ### ESN online training using the least-mean-square algorithm

# Another way to calculate $W^o$ is by altering the weights each time step. This is different from the above algorithm as it is done throughout the training period, rather than batch calculations using linear regression at the end.
# 
# The least-mean-square algorithm can be broken into two parts. The error function:
# 
# \begin{align*}
# e(k) = d(k) - x(k)\cdot \hat w(k)
# \end{align*}
# 
# Where:
# 
#  - $e(k)$ is the error value at step $k$
#  - $d(k)$ is the target at step $k$
#  - $x(k)$ is the connection values into the output from the hidden layer
#  - $\hat w(k)$ is the weight vector for the connections
# 
# Using the error function, the weights for the next step $\hat w(k+1)$ can be calculated with:
# 
# \begin{align*}
# \hat w(k+1)=\hat w(k) + \mu x(k)e(k)
# \end{align*}
# 
# Define the online version of an Echo State Network:

# In[1]:


def esn_online(Xtr, Xte, k, Nr=None, progress=True, a=0.95, mu=0.001):
    # Number of inputs
    Ni = 1

    # Number of hidden neurons
    if Nr is None:
        Nr = int(Xtr.shape[0] / 3)
    
    # Weights of inputs to neurons
    Wi = np.random.uniform(-1,1,(Nr, Ni))
    
    
    # Weights between neurons in the hidden layer
    Wr = np.random.uniform(-1, 1, (Nr, Nr))
    
    # set new weights between neurons in hidden layer
    Wr = a * Wr /  spectral_radius(Wr)

    x = np.zeros(Nr)
    Wo = np.zeros((Nr, 1))
    
        
    # itterate all steps and generate Xt for each
    for i in range(Xtr.shape[0] - k):
        if progress and i % 50 == 0:
            clear_output(wait=True)
            print("{:<8}{:>6.0%}".format("Training", i/Xtr.shape[0]))

        term1 = Wr @ x
        term2 = Wi @ Xtr[i]

        # Define the noise to be added to the network
        epsilon = np.random.normal(0, 0.1, size=Nr)
    
        x = np.tanh(term1 + term2 + epsilon)
        
        # calculate the error
        e = Xtr[i + k] - x@Wo
        
        #  itterativly adjust the output weights based on the error
        Wo += (mu*x*e).reshape(Nr, 1)

    Wo = Wo.T

    # Initiate testing
    x = np.zeros(Nr)
    Z = []
    
    # Go through testing and predict
    for i in range(Xte.shape[0] - k):
        if progress and i % 50 == 0:
            clear_output(wait=True)
            print("{:<8}{:>6.0%}".format("Testing", i/Xte.shape[0]))
        
        term1 = Wr @ x
        term2 = Wi @ Xte[i]

        # Define the noise to be added to the network
        epsilon = np.random.normal(0, 0.1, size=Nr)

        x = np.tanh(term1 + term2 + epsilon)

        Z.append(Wo@x)
    
    if progress:
        clear_output()

    return np.array(Z)


# Similar to the section above that used offline weight calculation, run the algorithm on the '2sin' dataset:

# In[14]:


k = 1
Z = esn_online(Xtr_2sin, Xte_2sin, k, progress=False)

plt.axis((-10, Xte_2sin[k:].shape[0] + 10, -3, 3))

plt.title("Echo state network predictions after trianing")
plt.plot(Xte_2sin[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)
plt.show()


# The results are really promising with the prediction line following almost identically to the provided targets. A note to be observed though is that it is unable to hit the peaks and troughs of the data. This could be caused by the averaging effect from continually updating the weights throughout the training phase.
# 
# For a side-by-side comparison, display offline vs. online methods:

# In[15]:


figure(figsize=(12,13))

k = 1
Z = esn(Xtr_2sin, Xte_2sin, k)

plt.subplot(221)
plt.title("ESN (offline) with k=1 step-ahead forecasting")
plt.axis((-10, Xte_2sin[k:].shape[0] + 10, -3, 3))
plt.plot(Xte_2sin[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)

k = 1
Z = esn_online(Xtr_2sin, Xte_2sin, k)

plt.subplot(222)
plt.title("ESN (online) with k=1 step-ahead forecasting")
plt.axis((-10, Xte_2sin[k:].shape[0] + 10, -3, 3))
plt.plot(Xte_2sin[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)

plt.show()


# Reitterating the point above, the online method is unable to reach the peaks seen by the batch process used by the offline method.
# 
# Concentrating on the least-mean-square algorithm, plot two values of $k$:

# In[16]:


figure(figsize=(12,13))

k = 1
Z = esn_online(Xtr_2sin, Xte_2sin, k)

plt.subplot(221)
plt.title("ESN model with k=1 step-ahead forecasting")
plt.axis((-10, Xte_2sin[k:].shape[0] + 10, -3, 3))
plt.plot(Xte_2sin[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)

k = 10
Z = esn_online(Xtr_2sin, Xte_2sin, k)

plt.subplot(222)
plt.title("ESN model with k=10 step-ahead forecasting")
plt.axis((-10, Xte_2sin[k:].shape[0] + 10, -3, 3))
plt.plot(Xte_2sin[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)

plt.show()


# Various take-aways from the above graphs can be made:
# 
# The averaging effect is more pronounced with a larger value of $k$, this could be a result of less information to what is next, as it is further in the future.
# 
# Not only scaling, but the actual line deviates from the pattern of the target data. In several places the prediction goes in the opposite direction to the testing data.
# 
# Overall, it is interesting to see that it is less able to emulate the '2sin' dataset.
# 
# To test these two algorithms on a different dataset, load the 'lorenz' file from the archive:

# In[18]:


X_lorenz = np.genfromtxt("./lorenz.txt", delimiter=",")

X_lorenz = np.array([np.array((x, )) for x in X_lorenz])

plt.title("The dataset lorenz")
plt.plot(X_lorenz[1:], "b")
plt.show()


# This dataset is an example of a Lorenz attractor, which is chaotic in nature.
# 
# Similar to in part 1, split the dataset into training and testing sets:

# In[19]:


Xtr_lorenz, Xte_lorenz = np.split(X_lorenz, 2)


# Compare the ability of the least-mean-square algorithm of both datasets:

# In[20]:


figure(figsize=(12,13))

k = 1
Z = esn_online(Xtr_2sin, Xte_2sin, k, Nr=300)

plt.subplot(221)
plt.title("ESN, k=1, 2sin dataset")
plt.axis((-10, Xte_2sin[k:].shape[0] + 10, -3, 3))
plt.plot(Xte_2sin[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)

k = 1
Z = esn_online(Xtr_lorenz, Xte_lorenz, k, Nr=300)

plt.subplot(222)
plt.title("ESN, k=1, lorenz dataset")
plt.axis((-10, Xte_lorenz[k:].shape[0] + 10, -20, 20))
plt.plot(Xte_lorenz[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)

plt.show()


# It is interesting to see that in both datasets the algorithm is able to closely emualte the target movements. Despite this, it suffers in both cases from averaging, resulting in an inability to reach the extremes of each respective dataset.
# 
# Here, we test the effect caused by various reservoir sizes:

# In[21]:


E = []
N = range(10, 250, 20)

for n in N:
    Z = esn_online(Xtr_lorenz, Xte_lorenz, k, progress=False, Nr=n)
    E.append(nrmse(Xte_lorenz[k:], Z))
    print("Nr={:<6} e={:<6.5}".format(n, E[-1]))


# In[22]:


plt.title("NRMSE observed for a range of $N_r$ values")
plt.plot(N, E, "r")
plt.xlabel("reservoir size")
plt.ylabel("NRMSE")
plt.show()


# In[23]:


Nr = N[np.argmin(E)]
print("Best Nr:", Nr)


# From the above experiment, it seems that in general the best reservoir is in the range of 100-150.
# 
# An interesting observation is if $k=100$:

# In[24]:


figure(figsize=(12,13))

k = 1
Z = esn_online(Xtr_lorenz, Xte_lorenz, k, Nr=Nr)

plt.subplot(221)
plt.title("ESN model with k=1 step-ahead forecasting")
plt.axis((-10, Xte_lorenz[k:].shape[0] + 10, -20, 20))
plt.plot(Xte_lorenz[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)

k = 100
Z = esn_online(Xtr_lorenz, Xte_lorenz, k, Nr=Nr)

plt.subplot(222)
plt.title("ESN model with k=100 step-ahead forecasting")
plt.axis((-10, Xte_lorenz[k:].shape[0] + 10, -20, 20))
plt.plot(Xte_lorenz[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)

plt.show()


# The neural network has detected that there are two clusters of values (the two areas of attraction in the Lorenz chaotic attractor), without specifically knowing which as it is not consistant. This is of interest as it demonstrates the error function locally optimizing that if the data is greater than zero it is likely to reverse at some point. The dual lane nature when $k=100$ reveals an aspect (albeit not very useful) of the dataset.
# 
# Similarly, observations can be made when $k=45$:

# In[25]:


figure(figsize=(12,13))

k = 1
Z = esn_online(Xtr_lorenz, Xte_lorenz, k, Nr=300)

plt.subplot(221)
plt.title("ESN model with k=1 step-ahead forecasting")
plt.axis((-10, Xte_lorenz[k:].shape[0] + 10, -20, 20))
plt.plot(Xte_lorenz[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)

k = 45
Z = esn_online(Xtr_lorenz, Xte_lorenz, k, Nr=300)

plt.subplot(222)
plt.title("ESN model with k=45 step-ahead forecasting")
plt.axis((-10, Xte_lorenz[k:].shape[0] + 10, -20, 20))
plt.plot(Xte_lorenz[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)

plt.show()


# In this case, the neural network has estimated that in the general case the data is centered around zero. At zero, it is likely to at least be in a local optima.
# 
# To get a better representation of the effect altering $k$ has, plot a graph:

# In[26]:


E = []
K = range(1, 50)

for k in K:
    Z = esn_online(Xtr_lorenz, Xte_lorenz, k, progress=False, Nr=Nr)
    E.append(nrmse(Xte_lorenz[k:], Z))
    print("k={:<6} e={:<6.5}".format(k, E[-1]))

plt.title("Error observed for a range of $k$ values")
plt.plot(K, E, "r")
plt.show()


# A very clear pattern emerges from the above graph. First, the optimal value is $k=1$ similar to in the '2sin' and offline-ESN combinations. Secondly, there is a clear dip centered around ~30. The dip can be potentially explained by the osscilating nature of the dataset, where it is seems to be osscilating at a frequency of ~30 steps. This osscilation is not uniform though, explaining the worse error value compared to closer forecasts.

# In[27]:


k = K[np.argmin(E)]
print("Best k:", k)


# In most executions of this notebook the best $k$ value is 1, although occasionally it is 2. I put this down to the random initialization of the network.
# 
# Putting $Nr$ and $k$ together, we can produce an optimal network to forecast the 'lorenz' dataset. With enough computing power it would make sense to test these variables together to form a map of error values, but these time constraints meant this was not realistic. The resulting variable pair should be fairly close though:

# In[28]:


Z = esn_online(Xtr_lorenz, Xte_lorenz, k, Nr=Nr)

plt.title("ESN model with k=5 step-ahead forecasting")
plt.axis((-10, Xte_lorenz[k:].shape[0] + 10, -20, 20))
plt.plot(Xte_lorenz[k:], "b", label="testing")
plt.plot(Z[:,0], "r", label="predictions")
legend(loc=1)

plt.show()


# In[ ]:





# In[ ]:




