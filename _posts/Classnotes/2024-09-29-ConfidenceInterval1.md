---
title: 'Confidence Interval 1'
date: 2024-09-29
categories: blog
tags:
    - class notes
    - statistics
---

This blog contains my class notes from the Mathematical Statistics with Application course. We’ll first start with a fundamental but important concept in statistics: **confidence interval**.

In real life, it’s almost impossible to measure a true value such as the average height of all humans. Instead, we can calculate an interval where the true parameter falls in with a desired level of probability, using only a sample obtained from the population.

## Definition
Let’s introduce some terms necessary for better understanding. 

Let $\theta$ denote the population parameter (i.e., mean, standard deviation). $\hat{\theta}$ is then the estimator used to estimate the population parameters.

It’s important to note that confidence intervals commonly rely on the assumption that the population follows a normal distribution (this is not always the case though). This assumption allows us to use *z-score* (or *t-score* and *chi-square score* in some other cases) to obtain the desire range of the estimator with a specified probability after normalizing the estimator.

![img]('/assets/classnote_photos/CI-photo1.png')

If the given population has a normal distribution, we can estimate $\theta$, using the z-score formula as follows:

$$ z = \frac{\hat{\theta} - \theta}{\sigma_{\hat{\theta}}} \rightarrow \theta = \hat{\theta} \plusmn z\times\sigma_{\hat{\theta}} (for two tails) $${: .notice}

![img](/assets/classnote_photos/CI-photo2.png)

That means to construct the interval we need to find the z-value and the standard error $\sigma_{\hat{\theta}}$. FYI, their product is called the *margin of error*.

Find z-value: With the given probability *p* of the interval, we calculate the area or the probability on the two tails by $\frac{1-p}{2}$ based on the fact that the normal curve is symmetric and the total probability under the curve is equal to 1. Using the Z table, we can find the z value corresponding to the area on the tail. For example, for a 95% confidence interval, we want a z-value that cuts off an area of 0.025 on each tail.

By the way, *p* is usually called confidence level denoted by *c* in the context of confidence interval.

Another thing is the standard error which represents the standard deviation of the sampling distribution. For instance, if we take a random sample and calculate its mean and repeat this process multiple times, the standard deviation of those sample means is the standard error. It indicates how much a sample mean is expected to vary from the true mean.

The standard error is the key ingredient in constructing confidence intervals. Its formula varies depending on the type of population parameter being estimated. Here are some common scenarios: 

|Parameter|Standard Error|
|-----|-----|
|Population mean| $\frac{\sigma}{\sqrt{n}}$ |
|Population proportion| $\sqrt{\frac{p(1-p)}{n}}$ |
|Difference between two population means| $\sqrt{\frac{\sigma_{1}^2}{n_{1}} + \frac{\sigma_{2}^2}{n_{2}}} $ |
|Difference between two proportion means| $\sqrt{\frac{p_{1}(1-p{1})}{n_{1}} + \frac{p_2(1-p_{2})}{n_{2}}}$ |
{: .notice--info}

Knowing how to find z-value and compute the standard error, we now can construct a confidence interval for the desired parameter.

## Interpretation
There are some common misunderstandings about how to correctly interpret the results of confidence intervals. For example, a 95% confidence interval does not mean that there is a 95% probability that the true parameter is contained in this interval, nor does it mean 95% of the sample data lie within the confidence interval. The reason is that the true parameter is a fixed value , whereas the confidence intervals vary depending on the sample statistic. 

The correct interpretation is: “The true parameter is in the confidence interval with 95% confidence”. In other words, when we were to construct 100 confidence intervals from different samples of the same population, approximately 95 of those intervals will contain the true population parameter.

## Summary
In this post, we’ve explored how to construct confidence intervals for a population parameter. However, this only works best under when the population has a normal distribution, or the sample size is sufficiently large. For smaller sample sizes, we need to approach a different method, which I’ll discuss in future post.
