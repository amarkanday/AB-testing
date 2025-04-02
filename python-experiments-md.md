# Designing, Running, and Analyzing Experiments in Python

**Author:** Ashish Markanday (Converted to Python by Claude)  
**Date:** April 1, 2025

## Table of Contents
1. [Introduction](#introduction)
2. [Common Distributions](#common-distributions)
3. [Parametric vs Non-parametric Tests](#parametric-vs-non-parametric-tests)
4. [General Workflow](#general-workflow)
5. [Independent Sample T-Test](#independent-sample-t-test)
6. [T-test Done the Right Way](#t-test-done-the-right-way)
7. [One-Way ANOVA](#one-way-anova)
8. [Factorial ANOVA](#factorial-anova)
9. [Generalized Linear Models](#generalized-linear-models)
10. [References](#references)

## Introduction

This document provides a comprehensive guide to designing, running, and analyzing experiments using Python. It covers various statistical methods from basic t-tests to more complex factorial ANOVA and generalized linear models. The examples focus on practical applications like A/B testing, task completion analysis, and user preference studies.

## Common Distributions

* **Normal**: Most common distribution, commonly called a Bell curve
* **Log normal**: Time taken to complete a task
* **Exponential distributions**: Wealth
* **Poisson**: Count data such as counts of rare events (integer response only)
* **Binomial distribution**: Binary outcomes like coin tosses
* **Power law distribution**: Distribution of friends in a social network
* **Beta distribution**: Used for proportions and probabilities

## Parametric vs Non-parametric Tests

* **Parametric tests** assume a specific distribution of data (Normal, Exponential, Poisson, etc.)
* **Non-parametric tests** do not make those assumptions. They typically work by comparing the rank of each item in one group with items in other groups

## General Workflow

1. Collect data
2. Explore data through descriptive statistics and visualizations
3. Check statistical assumptions
4. Run appropriate tests
5. Analyze and interpret results

## Independent Sample T-Test

### Example Case
You're launching a new design for a home page and want to see if it impacts user retention. You'll compare the new design with the existing one.

### Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import shapiro, levene, ttest_ind
```

### Loading and Exploring Data
```python
# Read in data
logins11Day = pd.read_csv("logins11Day.csv")
logins11Day['Site'] = logins11Day['Site'].astype('category')

# Summary statistics
print(logins11Day.groupby('Site')['Logins'].describe())
print(logins11Day.groupby('Site')['Logins'].agg(['mean', 'std']))

# Visualizations
plt.figure(figsize=(10, 6))
plt.hist(logins11Day[logins11Day['Site'] == 'Old']['Logins'], bins=10)
plt.title('Histogram of Logins on Old Site')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(logins11Day[logins11Day['Site'] == 'New']['Logins'], bins=10)
plt.title('Histogram of Logins on New Site')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Site', y='Logins', data=logins11Day)
plt.show()
```

### Running the T-test
```python
# Simple t-test without checking assumptions
t_stat, p_val = ttest_ind(
    logins11Day[logins11Day['Site'] == 'Old']['Logins'],
    logins11Day[logins11Day['Site'] == 'New']['Logins'],
    equal_var=True
)
print(f"T-test result: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
```

## T-test Done the Right Way

### Step 1: Import Data and Run Descriptive Statistics
```python
# Read data
anova1 = pd.read_csv("anova1.csv")
anova1['Subject'] = anova1['Subject'].astype('category')
anova1['TASK'] = anova1['TASK'].astype('category')

# Summary statistics
print(anova1.groupby('TASK')['Time'].describe())
print(anova1.groupby('TASK')['Time'].agg(['mean', 'std']))
```

### Step 2: Check Normality Assumption
```python
# Shapiro-Wilk normality test
stat, p_task1 = shapiro(anova1[anova1['TASK'] == 'Task1']['Time'])
print(f"Shapiro-Wilk test for Task1: statistic = {stat:.4f}, p-value = {p_task1:.4f}")

stat, p_task2 = shapiro(anova1[anova1['TASK'] == 'Task2']['Time'])
print(f"Shapiro-Wilk test for Task2: statistic = {stat:.4f}, p-value = {p_task2:.4f}")

# Check residuals normality
model = ols('Time ~ TASK', data=anova1).fit()
residuals = model.resid
stat, p_residuals = shapiro(residuals)
print(f"Shapiro-Wilk test for residuals: statistic = {stat:.4f}, p-value = {p_residuals:.4f}")

# QQ plot
plt.figure(figsize=(10, 6))
sm.qqplot(residuals, line='s')
plt.title('QQ Plot of Residuals')
plt.show()
```

### Step 3: Check Homoscedasticity Assumption
```python
# Levene's test
stat, p_levene = levene(
    anova1[anova1['TASK'] == 'Task1']['Time'], 
    anova1[anova1['TASK'] == 'Task2']['Time'],
    center='mean'
)
print(f"Levene's test: statistic = {stat:.4f}, p-value = {p_levene:.4f}")
```

### Step 4: Transform Data if Needed
```python
# Log transformation
anova1['logTime'] = np.log(anova1['Time'])

# Recheck assumptions
stat, p_task1_log = shapiro(anova1[anova1['TASK'] == 'Task1']['logTime'])
print(f"Shapiro-Wilk test for Task1 (log): statistic = {stat:.4f}, p-value = {p_task1_log:.4f}")

stat, p_task2_log = shapiro(anova1[anova1['TASK'] == 'Task2']['logTime'])
print(f"Shapiro-Wilk test for Task2 (log): statistic = {stat:.4f}, p-value = {p_task2_log:.4f}")

stat, p_levene_log = levene(
    anova1[anova1['TASK'] == 'Task1']['logTime'], 
    anova1[anova1['TASK'] == 'Task2']['logTime'],
    center='median'
)
print(f"Levene's test (log, median): statistic = {stat:.4f}, p-value = {p_levene_log:.4f}")
```

### Step 5: Run Appropriate Test
```python
# For normally distributed data with equal variances
t_stat, p_val = ttest_ind(
    anova1[anova1['TASK'] == 'Task1']['logTime'],
    anova1[anova1['TASK'] == 'Task2']['logTime'],
    equal_var=True
)
print(f"T-test result for logTime: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

# Non-parametric alternative
from scipy.stats import mannwhitneyu
u_stat, p_val = mannwhitneyu(
    anova1[anova1['TASK'] == 'Task1']['Time'],
    anova1[anova1['TASK'] == 'Task2']['Time']
)
print(f"Mann-Whitney U test: statistic = {u_stat:.4f}, p-value = {p_val:.4f}")
```

## One-Way ANOVA

### Step 1: Load and Examine Data
```python
# Read data with 3 tasks
anova2 = pd.read_csv("anova2.csv")
anova2['TASK'] = anova2['TASK'].astype('category')

# Summary statistics
print(anova2.groupby('TASK')['Time'].describe())
print(anova2.groupby('TASK')['Time'].agg(['mean', 'std']))
```

### Step 2: Check Assumptions and Transform if Needed
```python
# Apply log transformation
anova2['logTime'] = np.log(anova2['Time'])

# Check assumptions for transformed data
model2_log = ols('logTime ~ TASK', data=anova2).fit()
residuals2_log = model2_log.resid
stat, p_residuals2_log = shapiro(residuals2_log)
print(f"Shapiro-Wilk test for residuals (log): statistic = {stat:.4f}, p-value = {p_residuals2_log:.4f}")

# Test homoscedasticity
from scipy.stats import levene
stat, p_levene2_log = levene(
    anova2[anova2['TASK'] == 'Task1']['logTime'], 
    anova2[anova2['TASK'] == 'Task2']['logTime'],
    anova2[anova2['TASK'] == 'Task3']['logTime'],
    center='median'
)
print(f"Levene's test (log, median): statistic = {stat:.4f}, p-value = {p_levene2_log:.4f}")
```

### Step 3: Run ANOVA
```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# One-way ANOVA
model2_log = ols('logTime ~ TASK', data=anova2).fit()
anova_table = sm.stats.anova_lm(model2_log, typ=2)
print(anova_table)
```

### Step 4: Post-hoc Tests
```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tukey's HSD test for pairwise comparisons
tukey = pairwise_tukeyhsd(endog=anova2['logTime'], groups=anova2['TASK'], alpha=0.05)
print(tukey)

# Non-parametric alternative
from scipy.stats import kruskal
stat, p_val = kruskal(
    anova2[anova2['TASK'] == 'Task1']['Time'],
    anova2[anova2['TASK'] == 'Task2']['Time'],
    anova2[anova2['TASK'] == 'Task3']['Time']
)
print(f"Kruskal-Wallis test: statistic = {stat:.4f}, p-value = {p_val:.4f}")
```

## Factorial ANOVA

### Example Case
Understanding the impact of pricing on retention across different user engagement levels.

### Step 1: Load and Prepare Data
```python
pricing = pd.read_csv("pricing.csv")
pricing['Engagement'] = pd.Categorical(pricing['Engagement'], categories=['Low', 'High'], ordered=True)
pricing['Price'] = pd.Categorical(pricing['Price'], 
                                 categories=['Low (7.99)', 'Medium (9.99)', 'High (12.99)'], 
                                 ordered=True)
```

### Step 2: Explore Data
```python
# Summary statistics
print(pricing.groupby(['Engagement', 'Price'])['Days'].describe())

# Create interaction plot
fig = plt.figure(figsize=(10, 6))
means = pricing.groupby(['Price', 'Engagement'])['Days'].mean().unstack()
means.plot(marker='o', ax=fig.gca())
plt.title('Interaction Plot')
plt.ylabel('Days')
plt.grid(True)
plt.show()
```

### Step 3: Run Two-Way ANOVA
```python
# Two-way ANOVA
model = ols('Days ~ C(Price) + C(Engagement) + C(Price):C(Engagement)', data=pricing).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```

### Step 4: Post-hoc Tests
```python
from statsmodels.stats.multicomp import MultiComparison

# Comparisons for main effects
mc_price = MultiComparison(pricing['Days'], pricing['Price'])
result_price = mc_price.tukeyhsd()
print("Pairwise comparison for Price:")
print(result_price)

# For interaction terms
pricing['interaction'] = pricing['Engagement'] + ' & ' + pricing['Price']
mc_interaction = MultiComparison(pricing['Days'], pricing['interaction'])
result_interaction = mc_interaction.tukeyhsd()
print("Pairwise comparison for Interaction:")
print(result_interaction)
```

### Step 5: Non-Parametric Alternatives
```python
# Mann-Whitney U tests for each engagement level
p_values = []
test_pairs = []

for engagement in ['High', 'Low']:
    pairs = [
        ('Low (7.99)', 'Medium (9.99)'),
        ('Low (7.99)', 'High (12.99)'),
        ('Medium (9.99)', 'High (12.99)')
    ]
    
    for pair in pairs:
        u_stat, p_val = mannwhitneyu(
            pricing[(pricing['Engagement'] == engagement) & (pricing['Price'] == pair[0])]['Days'],
            pricing[(pricing['Engagement'] == engagement) & (pricing['Price'] == pair[1])]['Days']
        )
        p_values.append(p_val)
        test_pairs.append(f"{engagement} {pair[0]} vs {engagement} {pair[1]}")

# Apply Holm-Bonferroni correction
from statsmodels.stats.multitest import multipletests
reject, p_corrected, _, _ = multipletests(p_values, method='holm')
```

## Generalized Linear Models

GLMs extend Linear Models for cases with:
- Nominal or ordinal responses
- Non-normal response distributions (Poisson, exponential, gamma)

### Example 1: Multinomial Logistic Regression

#### Case: Analyzing User Preference by Sex
```python
# Read data
prefsABCsex = pd.read_csv("prefsABCsex.csv")
prefsABCsex['Subject'] = prefsABCsex['Subject'].astype('category')
prefsABCsex['Sex'] = prefsABCsex['Sex'].astype('category')

# Multinomial logistic regression
from statsmodels.discrete.discrete_model import MNLogit

# Prepare data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
prefsABCsex['Sex_num'] = le.fit_transform(prefsABCsex['Sex'])
dummies = pd.get_dummies(prefsABCsex['Pref'], prefix='pref')
y = dummies.iloc[:, :-1]  # Drop the last category as reference
X = sm.add_constant(prefsABCsex['Sex_num'])

# Fit model
mnlogit_model = MNLogit(y, X)
mnlogit_results = mnlogit_model.fit()
print(mnlogit_results.summary())

# Alternative: Chi-square test
contingency_table = pd.crosstab(prefsABCsex['Sex'], prefsABCsex['Pref'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square test: statistic = {chi2:.4f}, p-value = {p:.4f}, dof = {dof}")
```

#### Post-hoc Tests
```python
# Binomial tests for males
male_counts = contingency_table.loc['M']
male_total = male_counts.sum()

p_values_male = []
for pref in ['A', 'B', 'C']:
    count = male_counts[pref]
    p_val = stats.binom_test(count, male_total, p=1/3)
    p_values_male.append(p_val)

# Apply correction
reject, p_corrected, _, _ = multi.multipletests(p_values_male, method='holm')
```

### Example 2: Poisson Regression

#### Case: Error Counts in UI Design Versions
```python
# Read data
error_count = pd.read_csv("error_count.csv")
error_count['Design'] = pd.Categorical(error_count['Design'], 
                                      categories=['Orig', 'New1', 'New2'], 
                                      ordered=True)

# Fit Poisson regression model
import statsmodels.formula.api as smf
poisson_model = smf.glm(formula="Errors ~ Design", 
                       data=error_count, 
                       family=sm.families.Poisson()).fit()
print(poisson_model.summary())

# Incidence rate ratios
print("Incidence Rate Ratios (exp(coefficient)):")
print(np.exp(poisson_model.params[1:]))
```

## References

1. Kutner, M.H., Nachtsheim, C.J., Neter, J., & Li, W. (2005). Applied Linear Statistical Models (5th ed.). McGraw-Hill/Irwin.
2. McKinney, W. (2012). Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython. O'Reilly Media.
3. Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with python. In Proceedings of the 9th Python in Science Conference.
4. Virtanen, P., Gommers, R., Oliphant, T.E. et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17, 261–272.
