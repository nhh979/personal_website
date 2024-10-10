---
title: 'Confidence Interval for Small Samples'
date: 2024-10-06
categories: blog
tags:
    - class notes
    - statistics
---
 
Following the previous blog about confidence intervals for large samples, this post will discuss how to construct confidence intervals for small samples. 

For small samples, we need the assumption that the sample is randomly selected from a normal distribution. We will then use t-distributions rather than z-distributions. 

Similar to z-distributions, t-distributions also have the bell curve but with thicker tails. Additionally, the t-distribution depends on a parameter call **degrees of freedom (d.f.)** which determining the thickness of the tails.

![t-dist](https://raw.githubusercontent.com/nhh979/personal_website/refs/heads/master/assets/images/classnote_photos/t-dist.jpeg)


One important consideration is that this method cannot be used for proportion and difference in proportions for small sample sizes. The reason for this is that these estimators follow discrete binomial distributions which require a significantly large sample size to approximate a normal distribution.

Recall that $$\theta$$ denotes the population parameter, and $$\hat{\theta}$$ is the estimator. So $$\hat{\theta}$$ follows a t-distribution with mean $$\mu$$ and standard error $$\sigma_{\hat{\theta}}$$.

We can estimate $$\theta$$ with a confidence level $$c$$ using the t-statistic as follows:

$$T=\frac{\theta-\hat{\theta}}{\frac{s}{\sqrt{n}}} \rightarrow \theta = \hat{\theta} \pm t_{\frac{1-c}{2}} \sigma_{\hat{\theta}}$$ {: .notice}

with $$d.f.$$ degrees of freedom.

Similarly to the z-statistic, we can find the t-statistic using the t-distribution table with a given confidence level and known the degrees of freedom.

The standard error for $$\mu$$ can be computed easily using the formula:
$$\sigma_{\hat{\theta}} = \frac{s}{\sqrt{n}}$$


Calculating the standard error for $$\mu_1-\mu_2$$ is a bit more challenging, depending on whether the two population variances are assumed to be equal or not.
- If the two population variances are equal:

    $$\sigma_{\hat{\theta}} = s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$$  

    where $$s_p$$ is called the *pooled estimator* of the standard deviation and is computed by
    $$s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}$$
    with $$d.f. = n_1 + n_2 -2$$ degrees of freedom

- If the two population variances are not equal:

    $$\sigma_{\hat{\theta}} = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$$

    with $$d.f. = min(n_1-1, n_2-1)$$ degrees of freedom

## Coding Example
I'll create a data similar to the previous one but with a much smaller sample size and construct the confidence interval for the difference in mean grades between male and female students, using t-distribution with the assumption that the population variances of grades between male and female are equal.

Similar to the previous example, I first compute the mean grade for each gender and count the number of male and female students. I then construct the confidence interval for the difference in mean grade between male and female students, using t-distribution and the provided formulas.

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

# Create a data frame from the simulated data
df = pd.DataFrame({'Gender': gender, 'Percent_grade': grade_data})
print("The first 5 rows\n: ", df.head())

# Counts number of male and female students in the sample
counts = df.value_counts("Gender")
print("Student counts by gender: \n", counts)
# Calculate the mean percent grade across gender
mean_grade = df.groupby("Gender")['Percent_grade'].mean()
print('Mean grade by gender: \n', mean_grade)

# Calculate the standard deviation of grade
std_grade = df.groupby("Gender")['Percent_grade'].std(ddof=1) # ddof = 1 for the sample st. dev
print("Standard deviation of grade by gender: \n", std_grade)

# Extract the statistics values for male and female students
n_female = counts['Female']
n_male = counts["Male"]
mean_grade_female = mean_grade['Female']
mean_grade_male = mean_grade['Male']
std_grade_female = std_grade['Female']
std_grade_male = std_grade['Male']

# Compute the pooled estimator for the standard deviation
s_p = np.sqrt(((n_female-1)*std_grade_female**2 + (n_male-1)*std_grade_male**2) / (n_female + n_male - 2))
# Compute the standard error
se = s_p*np.sqrt(1/n_male + 1/n_female)

# Obtain t-score corresponding to the confidence level c and degrees of freedom
dof = n_female + n_male - 2  # Degrees of freedom
c = 0.95   # Confidence level 
t_score = np.abs(stats.t.ppf((1-c)/2, dof))

# Calculate the margin of error
E = t_score * se
# Construct the confidence interval
ci = (mean_grade_female - mean_grade_male - E, mean_grade_female - mean_grade_male + E)

# Display the results
print(f"The standard error: {se:.5f}")
print(f"The t-score with {c*100:.0f}% confidence level and {dof} degrees of freedoms: {t_score:.4f}")
print(f"The margin of error: {E:.4f}")
print(f"{c*100:.0f}% confidence interval for the difference in grade between male and female students: {ci}")
```

```
The first 5 rows
:     Gender  Percent_grade
0    Male      74.580813
1  Female      93.522365
2    Male      83.964449
3    Male      80.304547
4  Female      67.584588

Student counts by gender: 
 Gender
Male      24
Female    21
Name: count, dtype: int64

Mean grade by gender: 
 Gender
Female    76.223634
Male      75.803756
Name: Percent_grade, dtype: float64

Standard deviation of grade by gender: 
 Gender
Female    10.713323
Male       9.437874
Name: Percent_grade, dtype: float64

The standard error: 3.00339
The t-score with 95% confidence level and 43 degrees of freedoms: 2.0167
The margin of error: 6.0569
95% confidence interval for the difference in grade between male and female students: 
(-5.637031107147289, 6.476787613528183)
```

**Intepret:**  
We're 95% confident that the difference in mean grade between male and female students lies in the interval **(-5.637, 6.477)**.
Since 0 is contained in the interval, we can conclude that there is no difference in mean grade across gender.

## Summary
With small samples, we can construct confidence intervals for population mean and difference between means using t-distributions. Computing confidence intervals for difference between means is a little more challenging, depending on the assumption that the two population variances are equal or not. We also introduced a new term called *pooled estimator* of the standard deviation which combines standard deviations calculated from multiple samples into one overal standard deviation.