
However rather that inferring the distribution of methylation values associated to a CpG site from data for each year separately (or age binning), we turn our attention to modelling.
# Probabilistic model of epigenetic aging
In this section we will develop a probabilistic model of epigenetic ageing that allows us to assign a probability to methylation profile observed in an individual. We do this by inferring the probability distribution of methylation values in a cohort as a function of age. 

First let us set some notation. Methylation data associated to an individual, $j$, can be interpreted as a highly dimensional vector,

$$m_j = \left(m_j ^0, …., m_j^N, t_j \right),$$

where $m_j^i$ are the methylation levels of individual $j$ at site $i$, and $t_j$ is the age of the individual. We denote by $\mathcal{D}=\left \lbrace m_j \right \rbrace_{j \in J}$ the collection of all data points associated to a cohort of individuals. We further assume that all data points $m_j$ are independently distributed. Finally, we consider the cross-section of all data points $\mathcal{D}^i = \left \lbrace(m_j ^i, t_j)\right \rbrace_{j\in J}$, that is the data associated to a single CpG site for all individuals in the cohort.

## Single site model
We first proceed to develop the mathematical framework to infer the distribution of methylation values, in a single CpG site, as a function of age on a cohort level. This framework will then be easily scaled to include more CpG sites owing to the independence assumption of methylation observations. 

We now present two models of epigenetic ageing and will later discuss how to use them to select the optimal sites to use in a multisite model.

### Linear model

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

Then for a single individual $m_j$ , its associated probability for the single site model is

$$ P(m_j \mid \bar{a^i}, \bar{b^i},\bar{c^i}) = \mathcal{N}_{pdf}\left(m_j ^i; \bar{a^i}t_j+\bar{b^i}, \ \bar{c^i}\right). $$

### Drift model

As noted in the literature [REFERENCE!!!], there is a naturally occurring drift in methylation profile as a result of the stochastic nature of methylation gain. To accommodate for this process, we propose a drift model $\mathcal{M}_{drift}$ with a variance that increases linearly with time, as is the mean. Analogously to the linear model, the probability of observing a methylation value at site $i$ in a single individual $m_j$  is therefore equal to

$$
 P(m_j \mid \bar{a^i}, \bar{b^i},\bar{c^i}) = \mathcal{N}_{pdf}\left(m_j ^i; \bar{a^i}t_j+\bar{b^i}, \ \bar{c^i}t_j\right),  
$$

where $(\bar{a^i}, \bar{b^i},\bar{c^i})$ are, once again, the maximum likelihood estimator parameters associated to this model.

## Model comparison to filter saturating sites
In the second step of site selection, we measure relative evidence for each model in each site. As expected, most of the sites showed a very strong probability for the linear variance model. The constant variance model, $\mathcal{M}_{lin}$, not only is expected to be favoured in sites where the variance is constant in time, but also in saturating sites. This is because, in saturating sites, any linear increase in variance, due to drift, is halted by the effect of the saturation of methylation levels.
Since saturating sites cannot be used to accurately model epigenetic aging at old age, with the proposed linear models, we use model comparison techniques to filter saturating sites and only select those for which the drift model is favoured in further analyses.

### Akaike information criterion 

Associated to our maximum likelihood estimators for each model, $\mathcal{M}$, is its Akaike information criterion (AIC),

$$AIC_{\mathcal{M}} = 2k-\log \left(\hat{\mathcal{L}}_{\mathcal{M}}\right),$$

where $k$  is the number of parameters in model $\mathcal{M}$. 

To compare two models we compute RELATIVE AIC WEIGTHS BLABLABLA. AIC equations relative weights

-> $\omega ^i _{lin}, \omega ^i _{drift}$
We then create the subset of sites $I$ satisfying 
$$w^i _{drift}> 0.99, \ \forall i \in I.$$
## Multisite probabilistic model of epigenetic ageing

We now combine several sites to create more robust model of epigenetic aging

For an individual $m_j$, since methylation observations at each site are assumed to be independent, the combined probability of observing a methylation profile in sites $I$ is

$$
 P(m_j \mid \bar{a^i}, \bar{b^i},\bar{c^i}) = \prod_{i \in I}\mathcal{N}_{pdf}\left(m_j ^i; \bar{a^i}t_j+\bar{b^i}, \ \bar{c^i}t_j\right),  
$$

# Inferring an individual-specific changes in epigenetic aging

## Normalised speed of epigenetic ageing
We now use the multisite probability of epigenetic ageing to model the effect of an increase in the speed of epigenetic ageing.

In a single site, we define the average speed of ageing, $s_i$, as the time-derivative of the expectation of methylation values. In our drift model, since $\mathbb{E}\left[ m(t) \right] = \bar{a^i}t+\bar{b^i}$, the speed of ageing in this site is

$$s_i = \bar{a^i}.$$
We then model a normalised speed of epigenetic ageing (NSEA), $\alpha$, in a site by increasing the average speed of ageing. That is, the probability of observing a methylation value $m$ at age $t$ in an individual ageing at a normalised rate $\alpha$  is     

$$
P\left(m\mid t, \alpha \right) = \mathcal{N}_{pdf}\left(m_j ^i; \alpha \bar{a^i}t_j+\bar{b^i}, \ \bar{c^i}t_j\right).
$$

Notice that NSEA is normalised so that $\alpha= 1$ corresponds to the rate of epigenetic aging observed in the cohort. 

In the multisite model of epigenetic ageing we assuming that the NSEA is uniform across all sites and model the probability of observing an individual $m_j$ , conditional on a NSEA is

$$
 P(m_j \mid \alpha ) = \prod_{i \in I}\mathcal{N}_{pdf}\left(m_j ^i; \alpha\bar{a^i}t_j+\bar{b^i}, \ \bar{c^i}t_j\right).  
$$

Having fixed $(\bar{a^i}, \bar{b^i},\bar{c^i})$ in our previous steps, it is now easy to compute the likelihood of parameter $\alpha$, as

$$L(\alpha; m_j) = \prod_{i \in I}\mathcal{N}_{pdf}\left(m_j ^i; \alpha\bar{a^i}t_j+\bar{b^i}, \ \bar{c^i}t_j\right).  
$$
This allows us to compute the maximum a posteriori (MAP) estimator $\hat{\alpha}_j$, corresponding to the inferred NSEA for an individual.

Notice that the NSEA has been normalised so that $\alpha = 1$  corresponds to an individual whose rate of epigenetic ageing agrees with the average rate of the cohort. Correspondingly,  $\alpha= 2$  translates to an individual ageing twice and $\alpha= 1/2$ twice as slow as compared to the cohort average.
Furthermore, since NSEA modifies the average speed of aging $\bar{a}_i$, this increase in speed acts in the direction of the evolution of methylation as a function of time. That is, a positive NSEA returns higher methylation values on sites increasing with age and negative NSEA returns a lower values on decreasing, and the magnitude of the increase is proportional to the slope. See [@Fig:accelerated_invidividual] for an example.

## Epigenetic Bias
We now turn our attention to understand another phenomena naturally occurring in methylation studies. Either due to a technical artefacts resulting from the sequencing process or biological processes, the full methylation profile of an individual can be shifter. We term this effect as bias and model it as a further parameter in our model, $\beta$.

According to our drift model, global shifts of hypo or hyper methylation will result in a uniform modification of the expected methylation value of an individual at all sites. We therefore model the probability of observing a bias effect as  

$$
 P(m_j \mid \beta ) = \prod_{i \in I}\mathcal{N}_{pdf}\left(m_j ^i; \bar{a^i}t_j+\bar{b^i} + \beta, \ \bar{c^i}t_j\right).  
$$

Analogously to the NSEA case, we can now compute the likelihood of a bias, $b_j$ for an individual $m_j$  as

$$L(\beta_j; m_j) = \prod_{i \in I}\mathcal{N}_{pdf}\left(m_j ^i; \bar{a^i}t_j+\bar{b^i} + \beta_j, \ \bar{c^i}t_j\right),  $$
and obtain the MAP parameter $\bar{\beta}_j$ .

Notice that unlike NSEA, bias does not take into account the directionality of epigenetic ageing. A bias therefore translates into a uniform shift across all the sites affecting increasing and decreasing sites equally. See [@Fig:biased_individual] for an example.

![Example of a person that has a negative bias but no acceleration - he is globally shifted an equal value in both the increasing and decreasing sites. If one would correct for the shift, there would be no acceleration.](Figures/biased_individual.png){#fig:biased_individual}

## Combining NSEA and bias
We finally combine both individual-specific measures of ageing, NSEA and bias, as follows. The probability of observing and individual $m_j$  conditional on $\alpha$ and $\beta$  is

$$
P\left(m_j\mid \alpha, \beta \right) = \mathcal{N}_{pdf}\left(m_j ^i; \alpha \bar{a^i}t_j+\bar{b^i} + \beta, \ \bar{c^i}t_j\right).
$$

We can compute the likelihood associated to a combination of NSEA and bias to infer their posterior then infer, as usual, the MAP estimators $\bar{\alpha}_j$ and $\bar{\beta}_j$  that maximize the combined likelihood. 

- [ ] KDE Plot of alpha and beta 