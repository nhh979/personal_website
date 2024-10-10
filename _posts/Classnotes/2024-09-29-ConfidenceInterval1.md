---
title: 'Confidence Interval for Large Samples'
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

Let $$\theta$$ denote the population parameter (i.e., mean, standard deviation). $$\hat{\theta}$$ is then the estimator used to estimate the population parameters.

If the sample size is sufficiently large, the Central Limit Theorem states that the sampling distribution of the sample mean will be normally distributed. This allows us to use *z-score*  to compute the desire range of the estimator with a specified probability for parameters like population mean ($$\mu$$), population proportion $$\sigma$$, difference between means ($$\mu_1 - \mu_2$$), and difference between proportions ($$p_1 - p_2$$).

So, $$\hat{\theta}$$ is the estimator following a normal distribution with mean $$\theta$$ and standard error $$\sigma_{\hat{\theta}}$$. We can estimate $$\theta$$, using the z-score formula as follows:

$$ z = \frac{\hat{\theta} - \theta}{\sigma_{\hat{\theta}}} \rightarrow \theta = \hat{\theta} \pm z\times\sigma_{\hat{\theta}} ~(for~two~tails) $${: .notice}


That means to construct the interval we need to find the **z-value** and the **standard error** $$\sigma_{\hat{\theta}}$$. FYI, their product is called the *margin of error*.

<div class="container">
  <img src="https://raw.githubusercontent.com/nhh979/personal_website/refs/heads/master/assets/images/classnote_photos/CI-photo.png" alt="img" width="400" height="350">
</div>
  

To find the z-value for a given probability *p* of the interval, we calculate the area or the probability in the two tails by $$\frac{1-p}{2}$$ based on the fact that the normal curve is symmetric and the total probability under the curve is equal to 1. Using the Z table, we can find the z value corresponding to the area on the tail. For example, for a 95% confidence interval, we want a z-value that cuts off an area of 0.025 in each tail.

By the way, *p* is usually called confidence level denoted by *c* in the context of confidence interval.

The standard error represents the standard deviation of the sampling distribution. For instance, if we take a random sample and calculate its mean and repeat this process multiple times, the standard deviation of those sample means is the standard error. It indicates how much a sample mean is expected to vary from the true mean.


The standard error is the key ingredient in constructing confidence intervals. Its formula varies depending on the type of population parameter being estimated. Here are some common scenarios: 

|Parameter|Standard Error|
|-----|-----|
|$$\mu$$| $$\frac{\sigma}{\sqrt{n}}$$ |
|$$p$$| $$\sqrt{\frac{p(1-p)}{n}}$$ |
|$$\mu_1 - \mu_2$$| $$\sqrt{\frac{\sigma_{1}^2}{n_{1}} + \frac{\sigma_{2}^2}{n_{2}}} $$ |
|$$p_1 - p_2$$| $$\sqrt{\frac{p_{1}(1-p{1})}{n_{1}} + \frac{p_2(1-p_{2})}{n_{2}}}$$ |
{: .notice--info}

Knowing how to find z-value and compute the standard error, we now can construct a confidence interval for the desired parameter.

## Interpretation
There are some common misunderstandings about how to correctly interpret the results of confidence intervals. For example, a 95% confidence interval does not mean that there is a 95% probability that the true parameter is contained in this interval, nor does it mean 95% of the sample data lie within the confidence interval. The reason is that the true parameter is a fixed value , whereas the confidence intervals vary depending on the sample statistic. 

The correct interpretation is: “The true parameter is in the confidence interval with 95% confidence”. In other words, when we were to construct 100 confidence intervals from different samples of the same population, approximately 95 of those intervals will contain the true population parameter.

## Coding Example
I'll simulate a percentage grade data for both male and female students and construct a confidence interval for the difference in proportions of A grades across gender.

First off, I create 1000 grade samples using a truncated normal random variable. This ensures the data have a normal distribution shape with desired minimum and maximum values (0 and 100 respectively for grade). I then generate 1000 gender samples with the respective probability for each gender (0.4 for female, 0.6 for male).

Next, I put them all into a dataframe and convert the percent grade to GPA letter grades. After computing the number of male and female students as well as the proportions of A grade for each gender, I use the z-distribution to construct the confidence interval at a given confidence level for the difference between proportions.

```python
# Import neccesary libraries
import pandas as pd
import numpy as np
from scipy import stats

# Set a random seed for reproducibility
np.random.seed(42) 
# Set up some parameters for the simulated data
size = 1000             # sample size
mean_grade = 78         # mean of grades
std_grade = 10.2        # standard deviation of grades
min_grade = 0           # minimum grade
max_grade = 100         # maximum grade

# Calculate bounds for the truncated normal distribution
lower_bound = (min_grade - mean_grade) / std_grade
upper_bound = (max_grade - mean_grade) / std_grade

# Generate simulated data using truncated normal distribution
grade_data = stats.truncnorm.rvs(lower_bound, upper_bound, loc=mean_grade, scale=std_grade, size=size)

# Generate random gender data (40% female, 60% male)
gender = np.random.choice(['Female', 'Male'], replace=True, size=size, p=[0.4, 0.6])

# Create a function to conver percentage grade to GPA letter grade
def convert_to_GPA(x):
    if x >= 90:
        return 'A'
    elif x >= 80:
        return 'B'
    elif x >= 70:
        return 'C'
    elif x >= 60:
        return 'D'
    else:
        return 'F'

# Create a data frame from the simulated data
df = pd.DataFrame({'Gender': gender, 'Percent_grade': grade_data})
df['GPA'] = df['Percent_grade'].apply(convert_to_GPA)
print("The first 5 rows\n: ", df.head())

# Make a pivot table to count GPA grade across gender
gpa_counts = df.pivot_table(columns='Gender', index='GPA', aggfunc='count', margins=True, margins_name='Total Count')
gpa_counts.columns = gpa_counts.columns.droplevel(level=0)
print("GPA counts by gender: \n", gpa_counts)

# Calculate the grade proportions for each gender sample
prop_gpa_female = df.query('Gender == "Female"').value_counts('GPA', normalize=True)
print('\nProportion of grade for female: \n', prop_gpa_female)
prop_gpa_male= df.query('Gender == "Male"').value_counts('GPA', normalize=True)
print('\nProportion of grade for male: \n', prop_gpa_male)

# Extract the proportion of A grade and sample size for each gender
p_hat_male = prop_gpa_male['A']
p_hat_female = prop_gpa_female['A']
n_male = gpa_counts.loc['Total Count', 'Male']
n_female = gpa_counts.loc['Total Count', 'Female']

# Compute the standard error for the difference between proportions
se = np.sqrt(p_hat_male*(1-p_hat_male)/n_male + p_hat_female*(1-p_hat_female)/n_female)

# Obtain z-score corresponding to the given confidence level c
c = 0.90
z_score = np.abs(stats.norm.ppf((1-c)/2))

# Calculate the margin of error
E = z_score * se

# Construct the confidence interval
point_estimator = p_hat_male - p_hat_female     # Point estimator
ci = (point_estimator - E, point_estimator + E)

# Display the results
print(f'The A grade proportions of male and female students: {p_hat_male:.5f} and {p_hat_female:.5f}, respectively.')
print(f'The sample sizes of male and female students: {n_male} and {n_female}')
print(f"The standard error: {se:.5f}")
print(f"Z-score for {c*100:.0f}% confidence interval: {z_score:.3f}")
print(f"The margin of error: {E:.5f}")
print(f"{c*100:.0f}% confidence interval for the difference between grade proportions: {ci}")
```

```
The first 5 rows
:     Gender  Percent_grade GPA
0  Female      74.580813   C
1    Male      93.522365   A
2    Male      83.964449   B
3    Male      80.304547   B
4    Male      67.584588   D

GPA counts by gender: 
 Gender       Female  Male  Total Count
GPA                                   
A                41    72          113
B               111   180          291
C               137   221          358
D                73   122          195
F                23    20           43
Total Count     385   615         1000

Proportion of grade for female: 
 GPA
C    0.355844
B    0.288312
D    0.189610
A    0.106494
F    0.059740
Name: proportion, dtype: float64

Proportion of grade for male: 
 GPA
C    0.359350
B    0.292683
D    0.198374
A    0.117073
F    0.032520
Name: proportion, dtype: float64

The A grade proportions of male and female students: 0.11707 and 0.10649, respectively.
The sample sizes of male and female students: 615 and 385
The standard error: 0.02038
Z-score for 90% confidence interval: 1.645
The margin of error: 0.03352
90% confidence interval for the difference between grade proportions: (-0.02293768268954751, 0.04409701116594918)
```
**Intepret:**   
We're 90% confident that the difference in true proportions of A grade between male and female students lies in the interval **(-0.023, 0.044)**.
This interval contains 0, implying that there is no difference between the two proportions.

## Summary
In this post, we’ve explored how to construct confidence intervals for a population parameter. However, this only works best when the population has a normal distribution, or the sample size is sufficiently large. For smaller sample sizes, we need to approach a different method, which I’ll discuss in future post.
