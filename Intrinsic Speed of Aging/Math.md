However rather that inferring the distribution of methylation values associated to a CpG site from data for each year separately (or age binning), we turn our attention to modelling.

First let us set some notation. Methylation data associated to an individual, $j$, can be interpreted as a highly dimensional vector,

$$m_j = \left(m_j ^0, …., m_j^N, t_j \right),$$

where $m_j^i$ are the methylation levels of individual $j$ at site $i$, and $t_j$ is the age of the individual. We denote by $\mathcal{D}=\left \lbrace m_j \right \rbrace_{j \in J}$ the collection of all data points associated to a cohort of individuals. We further assume that all data points $m_j$ are independently distributed. Finally, we consider the cross-section of all data points $\mathcal{D}^i = \left \lbrace(m_j ^i, t_j)\right \rbrace_{j\in J}$, that is the data associated to a single CpG site for all individuals in the cohort.

# Single site linear model

A simple model, $\mathcal{M}_{lin}$, based on previously developed methylation clocks, assumes that the distribution of methylation values at age $t$  can be approximated by a normal distribution with a mean that increases linearly with time and a constant variance. That is, methylation values for CpG site $i$ are modelled by the random variable

$$
M^i(t) \sim \mathcal{N}\left(a^it+b^i, \ c^i\right)
$$

Therefore the probability of observing a methylation value $m$ at age $t$ in this site is

$$
P(m \mid t, a^i, b^i, c^i) = \mathcal{N}_{pdf}\left(m; \ a^it+b^i, \ c^i\right). 
$$

Since our data $\mathcal{D}$ is independently distributed, we can compute the probability of a observing the data conditional on  $\mathcal{M}_{lin}$ for a single site $i$ as

$$
P(\mathcal{D}^i \mid a^i, b^i, c^i) = \prod_{j \in J}\mathcal{N}_{pdf}\left(m_j^i; \ a^it_j+b^i, \ c^i\right). 
$$

We can then estimate the maximum likelihood estimator parameters $(\bar{a^i}, \bar{b^i},\bar{c^i})$ in the usual way by solving

$$\begin{align} P(\mathcal{D}^i \mid \bar{a^i}, \bar{b^i},\bar{c^i}) &= \max_{a^i, b^i, c^i} P(\mathcal{D}^i \mid a^i, b^i, c^i)\\
&= \hat{\mathcal{L}}_{lin}.
\end{align}$$

Then for a single individual $m_j$ , its associated probability will be

$$ P(m^i \mid \bar{a^i}, \bar{b^i},\bar{c^i}) = \mathcal{N}_{pdf}\left(m; \bar{a}^it+b^i, \ c^i\right). $$

# Single site drift model
- [ ] equations with changing variance


# Model comparison
Further, associated to our maximum likelihood estimator model is its Akaike information criterion (AIC)

$$AIC_{lin} = 2*3-\log \left(\hat{\mathcal{L}}_{lin}\right)$$

AIC equations relative weights



- we want to get a probability of an individual with a single site
	- in two models
- we do model comp to drop saturating sites
- now we can go forward with the subset of sites to get the individuals complete methylation probability
	- this is the product
	- this is modelling the age
- and then we can expand this with 
	- we want to model parameters specific to the person not the site like acceleration/bias
	- then you can look at the posterior distr to find optimal parameters 
