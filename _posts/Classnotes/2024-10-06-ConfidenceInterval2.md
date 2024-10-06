---
title: 'Confidence Interval for Small Samples'
date: 2024-10-06
categories: blog
tags:
    - class notes
    - statistics
---
 
Following the previous blog about confidence intervals for large samples, this post will discuss how to construct confidence intervals for small samples. 

For small samples, we need the assumption that the sample is randomly selected from a normal distribution. We will then use t-distributions rather than z-distributions. Similar to z-distributions, t-distributions also have the bell curve but with thicker tails. Additionally, the t-distribution depends on a parameter call **degrees of freedom (d.f.)** which determining the thickness of the tails.

[t-dist](https://raw.githubusercontent.com/nhh979/personal_website/refs/heads/master/assets/images/classnote_photos/t-dist.jpeg)


One important consideration is that this method cannot be used for proportion and difference in proportions for small sample sizes. The reason for this is that these estimators follow discrete binomial distributions which require a significantly large sample size to approximate a normal distribution.

Recall that $$\theta$$ denotes the population parameter, and $$\hat{\theta}$$ is the estimator. So $$\hat{\theta}$$ follows a t-distribution with mean $$\mu$$ and standard error $$\sigma_{\hat{\theta}}$$.

We can estimate $$\theta$$ with a confidence level $$c$$ using the t-statistic as follows:

$$T=\frac{\theta-\hat{\theta}}{\frac{s}{\sqrt{n}}} \rightarrow \theta = \hat{\theta} \pm t_{\frac{1-c}{2}} \sigma_{\hat{\theta}}$$ {: .notice}

with $$d.f.$$ degrees of freedom.

Similarly to the z-statistic, we can find the t-statistic using the t-distribution table with a given confidence level and known the degrees of freedom.

The standard error for $$\mu$$ can be computed easily using the formula:
$$\sigma_{\hat{\theta}} = \frac{s}{\sqrt{n}}$$
and $$d.f. = n-1$$ degrees of freedom for t-statistic

Calculating the standard error for $$\mu_1-\mu_2$$ is a bit more challenging, depending on the whether that the two population variances are assumed to be equal or not.
- If the two population variances are equal:

        $$\sigma_{\hat{\theta}} = s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$$  

        where $$s_p$$ is called the *pooled estimator* of the standard deviation and is computed by
        $$s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}$$
        with $$d.f. = n_1 + n_2 -2$$ degrees of freedom

- If the two population variances are not equal:

        $$\sigma_{\hat{\theta}} = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$$
        
        with $$d.f. = min(n_1-1, n_2-1)$$ degrees of freedom

## Summary
With small samples, we can construct confidence intervals for population mean and difference between means using t-distributions. Computing confidence intervals for difference between means is a little more challenging, depending on the assumption that the two population variances are equal or not. We also introduced a new term called *pooled estimator* of the standard deviation which combines standard deviations calculated from multiple samples into one overal standard deviation.