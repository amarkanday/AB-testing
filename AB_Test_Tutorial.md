
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, wilcoxon, mannwhitneyu, kruskal
from scipy.stats import lognorm, ks_2samp, poisson, binom
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import statsmodels.stats.multitest as multi
from sklearn.preprocessing import LabelEncoder
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import lilliefors
```

## Common distributions 

* Normal: Most common for distribution, commonly called a Bell curve. 
* Log normal: Time taken to complete a task
* Exponential distributions: Wealth
* Poisson: Count data such as counts of rare events. This is only integer response.
* Binomial distribution: Coin toss 
* Power law distribution: Distribution of friends in a social network
* Beta distribution

----

## Parametric Vs Non parametric tests  

* Parametric assume a distribution of data such a Normal, exponential, Poisson etc.
* Non parametric tests do not make those assumptions. They typically work by comparing the rank of each item in a variant with each item in the other variants 

----

## General workflow 

1. Collect data
2. Perform exploratory data analysis
3. Check statistical assumptions
4. Choose appropriate statistical test
5. Run the test and analyze results

----

### Factor 2 Level test: Independent Sample T Test 

Example: You are launching a new design for Apollo home, and you want to see if that has an impact on user retention. There is only one redesign (one variant) and you will be comparing that with existing home page.

#### Background: 

* Lets call the new redesign as "New" and old design as "Old"
* You have done your due diligence and selected a metric that will track retention. Lets call this metric as App logins in the next 11 days 
* You have done your sample size calculations and decided to send 50% of new users to the new design and 50% to the older design 

#### Analysis

Descriptive test

* Summary statistics such as mean, median, quantile etc.
* Plot the distribution / histogram of the response variable
* Plot box plots
* Visual examination of the plots

* Do you think the responses look different
    + Run statistical test 

#### Read input file 

```python
# Independent-samples t-test

# Read in a data file with page views from an A/B test
logins11Day = pd.read_csv("logins11Day.csv")
print(logins11Day.head())
logins11Day['Site'] = logins11Day['Site'].astype('category')  # Convert to nominal factor
print(logins11Day.describe())
```

#### Run descriptive statistics

```python
# Descriptive statistics by Site
site_groups = logins11Day.groupby('Site')
print(site_groups['Logins'].describe())
print(site_groups['Logins'].agg(['mean', 'std']))
```

#### Plot distribution of data 

* Notice that the distribution for the second graph is not quite normal 
    + We will ignore this for now, and revisit this in the next example 
    
```python
# Graph histograms and a boxplot
plt.figure(figsize=(10, 6))
plt.hist(logins11Day[logins11Day['Site'] == 'Old']['Logins'], bins=10)
plt.title("Histogram of Logins on Old Site")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(logins11Day[logins11Day['Site'] == 'New']['Logins'], bins=10)
plt.title("Histogram of Logins on New Site")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Site', y='Logins', data=logins11Day)
plt.title("Boxplot of Logins by Site")
plt.show()
```

#### Run t-test 

* We are running the default t-test assuming that variances are equal 
    + I am highlighting this example because this is not a prescribed way of running t-tests
    + We have not checked any Anova assumptions

```python
# Independent-samples t-test
t_stat, p_val = ttest_ind(
    logins11Day[logins11Day['Site'] == 'Old']['Logins'],
    logins11Day[logins11Day['Site'] == 'New']['Logins'],
    equal_var=True
)
print(f"T-test result: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
```

Test says that the 2 groups are __different__, but can we trust it?

------

### T-test done the right way: 1 Factor 2 Level test

Example: You have assigned users tasks (say completion of a survey or a time taken to complete an adventure) and you want to see if there is a difference between time taken each for task variant 

Background: 

* Let us compare the time taken to complete 2 tasks and call these task 1 and task 2. This is similar to the experiment in the previous example. We will add more tasks later. 
* For now just consider 1 factor, 2 variant test 
* You have done your sample size calculations and decided to send 50% of new users to task 1 and 50% to task 2

__Import data__ 

```python
# Read in a data file with task completion times (min) for 2 different tasks 
anova1 = pd.read_csv("anova1.csv")
print(anova1.head())
anova1['Subject'] = anova1['Subject'].astype('category')  # Convert to nominal factor
anova1['TASK'] = anova1['TASK'].astype('category')  # Convert to nominal factor
print(anova1.describe())
```

__Run descriptive tests__

_Summary stats suggest that time taken to complete the 2 tasks is different_

```python
# View descriptive statistics by TASK
task_groups = anova1.groupby('TASK')
print(task_groups['Time'].describe())
print(task_groups['Time'].agg(['mean', 'std']))
```

__Graph histograms and box plots__

_Histogram of the data is not really normal_

```python
# Graph histograms and a boxplot
plt.figure(figsize=(10, 6))
plt.hist(anova1[anova1['TASK'] == 'Task1']['Time'], bins=10)
plt.title("Histogram of Time to complete Task 1")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(anova1[anova1['TASK'] == 'Task2']['Time'], bins=10) 
plt.title("Histogram of Time to complete Task 2")
plt.show()
```

_Box plots suggest that there may be a difference between time taken_

```python
# Display information about the DataFrame
print(anova1.info())

# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='TASK', y='Time', data=anova1)
plt.title("Boxplot of Time by Task")
plt.show()
```

__Running an independent-samples t-test is not suitable because we have not tested Anova assumptions__

__Testing ANOVA assumptions__

* We will use Shapiro Wilk test to test assumptions for normality
   
```python
# Shapiro-Wilk normality test on response
stat, p_task1 = shapiro(anova1[anova1['TASK'] == 'Task1']['Time'])
print(f"Shapiro-Wilk test for Task1: statistic = {stat:.4f}, p-value = {p_task1:.4f}")

stat, p_task2 = shapiro(anova1[anova1['TASK'] == 'Task2']['Time']) 
print(f"Shapiro-Wilk test for Task2: statistic = {stat:.4f}, p-value = {p_task2:.4f}")
```

Notice that the P values are less than 5% for both test. This means that Time is not normally distributed.
We have violated the first assumption for Anova that the data should be normally distributed.

__Really what we are looking for is that the residuals are normally distributed__

```python
# Fit model and test residuals normality
model = ols('Time ~ TASK', data=anova1).fit()
residuals = model.resid

stat, p_residuals = shapiro(residuals)
print(f"Shapiro-Wilk test for residuals: statistic = {stat:.4f}, p-value = {p_residuals:.4f}")
```

We can also plot the qq plot to see if the residuals are normally distributed 

```python
# QQ plot of residuals
plt.figure(figsize=(10, 6))
sm.qqplot(residuals, line='s')
plt.title("QQ Plot of Residuals")
plt.show()
```

_QQ plot shows that the data is not normally distributed_

__Since the residuals are not normally distributed, you have 2 options__

* Use data transformations to make data normal and use parametric tests 
* Use a non parametric test. Non parametric tests do not assume a distribution 

__Continuing with a parametric test__

Typically when one is looking at task completion times, lognormal is a good candidate distribution to start.
Intuitive reasoning behind using a log normal distribution is that the "Log Normal" is the log of the normal distribution.
When users are asked to complete a task, majority will complete it with a certain time and a 
small minority will take much longer to complete the same task. That distribution generally follows a 
log normal distribution.
If log normal does not work, other distribution to try would be a poisson distribution, or a beta distribution.

**_T Test and Anova are generally robust and will work even if the data in not quite normal_**

* Kolmogorov-Smirnov test for log-normality
* Fit the distribution to a lognormal to estimate fit parameters
* Then supply those to a K-S test with the lognormal distribution fn

```python
# Test for log-normality for Task1
data_task1 = anova1[anova1['TASK'] == 'Task1']['Time']
shape, loc, scale = lognorm.fit(data_task1, floc=0)  # Fit lognormal distribution
ks_stat, p_value = ks_2samp(data_task1, lognorm.rvs(s=shape, loc=loc, scale=scale, size=1000))
print(f"KS test for lognormal fit (Task1): statistic = {ks_stat:.4f}, p-value = {p_value:.4f}")

# Test for log-normality for Task2
data_task2 = anova1[anova1['TASK'] == 'Task2']['Time']
shape, loc, scale = lognorm.fit(data_task2, floc=0)  # Fit lognormal distribution
ks_stat, p_value = ks_2samp(data_task2, lognorm.rvs(s=shape, loc=loc, scale=scale, size=1000))
print(f"KS test for lognormal fit (Task2): statistic = {ks_stat:.4f}, p-value = {p_value:.4f}")
```

P values greater than 5% indicates that the new fit does not violate the log normal assumptions

__Testing for the 2nd Anova assumption: Homoscedasticity (homogeneity of variance)__

Homogeneity of variance means that the variance does not increase as the size of the variable increases.
It is mostly seen in time series where the value in the this time period depends on value in the previous 
time period. In that case, errors from previous value get added to the current value and compounded. 
Other places where you would see that is where the relationship with the dependent variable is logarithmic. 
In those cases, box-cox transformation can help you alleviate heteroscedasticity.

__A recommended and a generic approach is to use GLM models and specify the distributions while modeling the data__

__We will touch on GLM models later in the document__

```python
# Test for homoscedasticity (homogeneity of variance)
stat, p_levene = levene(
    anova1[anova1['TASK'] == 'Task1']['Time'],
    anova1[anova1['TASK'] == 'Task2']['Time'],
    center='mean'  # Levene's test
)
print(f"Levene's test: statistic = {stat:.4f}, p-value = {p_levene:.4f}")

# Brown-Forsythe test
stat, p_bf = levene(
    anova1[anova1['TASK'] == 'Task1']['Time'],
    anova1[anova1['TASK'] == 'Task2']['Time'],
    center='median'  # Brown-Forsythe test
)
print(f"Brown-Forsythe test: statistic = {stat:.4f}, p-value = {p_bf:.4f}")
```

We run tests for both Mean and Median. 
Since the P value is less than 5%, this suggests that it does not satisfy homogeneity of variance assumption

**_You can use Welch's T test if the homoscedasticity assumption is violated_**

```python
# Welch t-test for unequal variances handles
# the violation of homoscedasticity, but not
# the violation of normality.
t_stat, p_val = ttest_ind(
    anova1[anova1['TASK'] == 'Task1']['Time'],
    anova1[anova1['TASK'] == 'Task2']['Time'],
    equal_var=False  # Welch t-test
)
print(f"Welch's t-test result: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
```

__Running the T test again, after controlling for Anova assumptions__

Steps
* Transform variables 
* Test for normality 
* Test for homoscedasticity
* Run appropriate T test 

Data transformation 

```python
# Create a new column in anova1 defined as log(Time)
anova1['logTime'] = np.log(anova1['Time'])  # Log transform
print(anova1.head())  # Verify
```

Visually examine transformed data 

```python
# Plot histograms for log-transformed data
plt.figure(figsize=(10, 6))
plt.hist(anova1[anova1['TASK'] == 'Task1']['logTime'], bins=10)
plt.title("Histogram of log(Time) to complete Task 1")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(anova1[anova1['TASK'] == 'Task2']['logTime'], bins=10)
plt.title("Histogram of log(Time) to complete Task 2") 
plt.show()

# Create boxplot for log-transformed data
plt.figure(figsize=(10, 6))
sns.boxplot(x='TASK', y='logTime', data=anova1) 
plt.title("Boxplot of log(Time) by Task")
plt.show()
```

Re-test for normality

```python
# Shapiro-Wilk normality test on log-transformed response
stat, p_task1_log = shapiro(anova1[anova1['TASK'] == 'Task1']['logTime'])
print(f"Shapiro-Wilk test for Task1 (log): statistic = {stat:.4f}, p-value = {p_task1_log:.4f}")

stat, p_task2_log = shapiro(anova1[anova1['TASK'] == 'Task2']['logTime'])
print(f"Shapiro-Wilk test for Task2 (log): statistic = {stat:.4f}, p-value = {p_task2_log:.4f}")
```

Re-test for normality of residuals 

```python
# Fit model for log-transformed data and test residuals
model_log = ols('logTime ~ TASK', data=anova1).fit()
residuals_log = model_log.resid

stat, p_residuals_log = shapiro(residuals_log)
print(f"Shapiro-Wilk test for residuals (log): statistic = {stat:.4f}, p-value = {p_residuals_log:.4f}")
```

Visually examine residuals 

```python
# QQ plot of residuals for log-transformed data
plt.figure(figsize=(10, 6))
sm.qqplot(residuals_log, line='s')
plt.title("QQ Plot of Residuals (log transform)")
plt.show()
```

Re-test for homoscedasticity

```python
# Test for homoscedasticity of log-transformed data
stat, p_levene_log = levene(
    anova1[anova1['TASK'] == 'Task1']['logTime'],
    anova1[anova1['TASK'] == 'Task2']['logTime'],
    center='median'  # Brown-Forsythe test
)
print(f"Brown-Forsythe test (log): statistic = {stat:.4f}, p-value = {p_levene_log:.4f}")
```

__Finally: Independent-samples t-test (now suitable for logTime)__

```python
# T-test for log-transformed data
t_stat, p_val = ttest_ind(
    anova1[anova1['TASK'] == 'Task1']['logTime'],
    anova1[anova1['TASK'] == 'Task2']['logTime'],
    equal_var=True
)
print(f"T-test result for logTime: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
```

**Running a non-parametric test**

_If the underlying distribution is not normal, and none of the transformations that you have tried to make the data normal seem to work, it is time to run a non-parametric test_

Non-parametric tests do not assume an underlying distribution.
Instead, they compare and rank each row element in the first 
group with other row elements in second group. 

When reporting results for non-parametric tests, you should report the median 
and not the mean.

```python
# Mann-Whitney U test (equivalent to Wilcoxon rank-sum test)
u_stat, p_val = mannwhitneyu(
    anova1[anova1['TASK'] == 'Task1']['Time'],
    anova1[anova1['TASK'] == 'Task2']['Time']
)
print(f"Mann-Whitney U test: statistic = {u_stat:.4f}, p-value = {p_val:.4f}")

# Also run on log-transformed data
u_stat, p_val = mannwhitneyu(
    anova1[anova1['TASK'] == 'Task1']['logTime'],
    anova1[anova1['TASK'] == 'Task2']['logTime']
)
print(f"Mann-Whitney U test (log): statistic = {u_stat:.4f}, p-value = {p_val:.4f}")
```

__This completes the section for running a one factor two level t test__
__In the next section we explore how to run one factor multiple levels Anova test__

---

### One factor multiple levels parametric one way Anova between populations

> We will add one more task to the previous dataset so that now we are comparing 3 tasks 

__One-way ANOVA__

* Read in a data file with task completion times (min) now from 3 tasks

```python
# Read data file with 3 tasks
anova2 = pd.read_csv("anova2.csv")
print(anova2.head())
anova2['TASK'] = anova2['TASK'].astype('category')  # Convert to nominal factor
print(anova2.describe())
```

* View descriptive statistics by TASK

```python
# Descriptive statistics by TASK
task_groups = anova2.groupby('TASK')
print(task_groups['Time'].describe())
print(task_groups['Time'].agg(['mean', 'std']))
```

* Explore new response distribution

```python
# Plot histograms for each task
plt.figure(figsize=(10, 6))
plt.hist(anova2[anova2['TASK'] == 'Task1']['Time'], bins=10)
plt.title("Histogram of Time to complete Task 1")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(anova2[anova2['TASK'] == 'Task2']['Time'], bins=10)
plt.title("Histogram of Time to complete Task 2")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(anova2[anova2['TASK'] == 'Task3']['Time'], bins=10)
plt.title("Histogram of Time to complete Task 3")
plt.show()
```

* Test normality for new TASK

```python
# Shapiro-Wilk normality test for Task3
stat, p_task3 = shapiro(anova2[anova2['TASK'] == 'Task3']['Time'])
print(f"Shapiro-Wilk test for Task3: statistic = {stat:.4f}, p-value = {p_task3:.4f}")

# Fit model and test residuals normality
model2 = ols('Time ~ TASK', data=anova2).fit()
residuals2 = model2.resid

stat, p_residuals2 = shapiro(residuals2)
print(f"Shapiro-Wilk test for residuals: statistic = {stat:.4f}, p-value = {p_residuals2:.4f}")

# QQ plot of residuals
plt.figure(figsize=(10, 6))
sm.qqplot(residuals2, line='s')
plt.title("QQ Plot of Residuals")
plt.show()
```

* Test log-normality of new TASK

```python
# Test for log-normality for Task3
data_task3 = anova2[anova2['TASK'] == 'Task3']['Time']
shape, loc, scale = lognorm.fit(data_task3, floc=0)  # Fit lognormal distribution
ks_stat, p_value = ks_2samp(data_task3, lognorm.rvs(s=shape, loc=loc, scale=scale, size=1000))
print(f"KS test for lognormal fit (Task3): statistic = {ks_stat:.4f}, p-value = {p_value:.4f}")
```

* Compute new log(Time) column and re-test

```python
# Create log-transformed column
anova2['logTime'] = np.log(anova2['Time'])
print(anova2.head())  # Verify

# Test normality of log-transformed data for Task3
stat, p_task3_log = shapiro(anova2[anova2['TASK'] == 'Task3']['logTime'])
print(f"Shapiro-Wilk test for Task3 (log): statistic = {stat:.4f}, p-value = {p_task3_log:.4f}")

# Fit model for log-transformed data and test residuals
model2_log = ols('logTime ~ TASK', data=anova2).fit()
residuals2_log = model2_log.resid

stat, p_residuals2_log = shapiro(residuals2_log)
print(f"Shapiro-Wilk test for residuals (log): statistic = {stat:.4f}, p-value = {p_residuals2_log:.4f}")

# QQ plot of residuals for log-transformed data
plt.figure(figsize=(10, 6))
sm.qqplot(residuals2_log, line='s')
plt.title("QQ Plot of Residuals (log transform)")
plt.show()
```

* Test homoscedasticity

```python
# Test for homoscedasticity of log-transformed data
stat, p_levene2_log = levene(
    anova2[anova2['TASK'] == 'Task1']['logTime'],
    anova2[anova2['TASK'] == 'Task2']['logTime'],
    anova2[anova2['TASK'] == 'Task3']['logTime'],
    center='median'  # Brown-Forsythe test
)
print(f"Brown-Forsythe test (log): statistic = {stat:.4f}, p-value = {p_levene2_log:.4f}")
```

**_One-way ANOVA, suitable now to logTime_**

```python
# One-way ANOVA for log-transformed data
model2_log = ols('logTime ~ TASK', data=anova2).fit()
anova_table = sm.stats.anova_lm(model2_log, typ=2)
print(anova_table)
```

```
How to analyze the output?
What is a F score?
```

**_Results of Anova tell us that at least one of the task time is different than the rest, however it does not tell which task time is different_**

> Next step is to do a pair-wise comparison of task times

```python
# Boxplot of Time by TASK
plt.figure(figsize=(10, 6))
sns.boxplot(x='TASK', y='Time', data=anova2)
plt.title("Boxplot of Time by Task")
plt.show()

# Tukey's HSD post-hoc test for pairwise comparisons
tukey = pairwise_tukeyhsd(endog=anova2['logTime'], groups=anova2['TASK'], alpha=0.05)
print(tukey)
```


* Task2 is different from Task1 and Task3
* Task1 and Task3 are similar

#### Non parametric anova 

```python
# Kruskal-Wallis test (non-parametric one-way ANOVA)
stat, p_val = kruskal(
    anova2[anova2['TASK'] == 'Task1']['Time'],
    anova2[anova2['TASK'] == 'Task2']['Time'],
    anova2[anova2['TASK'] == 'Task3']['Time']
)
print(f"Kruskal-Wallis test: statistic = {stat:.4f}, p-value = {p_val:.4f}")

# For log-transformed data
stat, p_val = kruskal(
    anova2[anova2['TASK'] == 'Task1']['logTime'],
    anova2[anova2['TASK'] == 'Task2']['logTime'],
    anova2[anova2['TASK'] == 'Task3']['logTime']
)
print(f"Kruskal-Wallis test (log): statistic = {stat:.4f}, p-value = {p_val:.4f}")

# Manual post-hoc Mann-Whitney U pairwise comparisons
task_pairs = [
    ('Task1', 'Task2'),
    ('Task1', 'Task3'),
    ('Task2', 'Task3')
]

p_values = []
for pair in task_pairs:
    u_stat, p_val = mannwhitneyu(
        anova2[anova2['TASK'] == pair[0]]['Time'],
        anova2[anova2['TASK'] == pair[1]]['Time']
    )
    p_values.append(p_val)
    print(f"Mann-Whitney U test ({pair[0]} vs {pair[1]}): u={u_stat:.4f}, p={p_val:.4f}")

# Apply Holm-Bonferroni correction for multiple comparisons
reject, p_corrected, _, _ = multi.multipletests(p_values, method='holm')
print("Corrected p-values (Holm method):")
for i, pair in enumerate(task_pairs):
    print(f"{pair[0]} vs {pair[1]}: p={p_corrected[i]:.4f}, reject H0: {reject[i]}")
```


### Factorial Anova (Crossed factor design)

> Suppose you want to understand the impact of pricing on retention for existing users. Also, you want to test it across different user engagement levels

---

* There are 3 pricing levels - low (7.99), medium (10.99) and high (12.99)
* There are 2 levels of user engagement for existing users - low and high. For this example, we assume the engagement levels definition is already established 

---

_Note: If you wanted to run an experiment that includes new users also, that would just be a separate "One factor [Price] multiple levels [low, medium and high] one way Anova test between populations", since we don't yet know the engagement level of new users_


__What questions are you trying to answer__

* Does pricing affect retention?
* Does user engagement level affect retention?
* Is there an interaction effect between pricing and user engagement level? For example: Are highly engaged users insensitive to price?


```python
# Read data file
pricing = pd.read_csv("pricing.csv")
pricing['Engagement'] = pd.Categorical(pricing['Engagement'], categories=['Low', 'High'], ordered=True)
pricing['Price'] = pd.Categorical(pricing['Price'], 
                                 categories=['Low (7.99)', 'Medium (9.99)', 'High (12.99)'], 
                                 ordered=True)

print(pricing.head())
print(pricing.info())
print(pricing.describe())

# Summary statistics by Engagement and Price
engagement_price_groups = pricing.groupby(['Engagement', 'Price'])
print(engagement_price_groups['Days'].describe())
print(engagement_price_groups['Days'].agg(['mean', 'std']))

# Create histograms for each combination of Engagement and Price
for engagement in ['High', 'Low']:
    for price in ['Low (7.99)', 'Medium (9.99)', 'High (12.99)']:
        plt.figure(figsize=(10, 6))
        plt.hist(pricing[(pricing['Engagement'] == engagement) & 
                        (pricing['Price'] == price)]['Days'], bins=10)
        plt.title(f"Histogram of unique app logins in 1st week after signup\n" +
                 f"Engagement = {engagement} & Price = {price}")
        plt.show()

# Boxplots for main effects
plt.figure(figsize=(10, 6))
sns.boxplot(x='Engagement', y='Days', data=pricing)
plt.title("Boxplot of Days by Engagement")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Price', y='Days', data=pricing)
plt.title("Boxplot of Days by Price")
plt.show()

# Boxplot showing interaction
plt.figure(figsize=(12, 6))
sns.boxplot(x='Price', y='Days', hue='Engagement', data=pricing)
plt.title("Boxplot for unique app logins for price and user engagement")
plt.show()

# Interaction plot
fig, ax = plt.subplots(figsize=(10, 6))
for name, group in pricing.groupby('Engagement'):
    ax.plot(group.groupby('Price')['Days'].mean().index, 
            group.groupby('Price')['Days'].mean(), 
            marker='o', label=name)
ax.set_xlabel('Price')
ax.set_ylabel('Days')
ax.set_title('Interaction Plot')
ax.legend()
plt.grid(True)
plt.show()

# Two-way ANOVA
model = ols('Days ~ C(Price) + C(Engagement) + C(Price):C(Engagement)', data=pricing).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Check residuals
plt.figure(figsize=(10, 6))
plt.hist(model.resid, bins=10, color='darkgray')
plt.title('Histogram of Residuals')
plt.show()

# Test normality of residuals
stat, p_residuals = shapiro(model.resid)
print(f"Shapiro-Wilk test for residuals: statistic = {stat:.4f}, p-value = {p_residuals:.4f}")

# Post-hoc tests for main effects
# For Engagement
mc_engagement = MultiComparison(pricing['Days'], pricing['Engagement'])
result_engagement = mc_engagement.tukeyhsd()
print("Pairwise comparison for Engagement:")
print(result_engagement)

# For Price
mc_price = MultiComparison(pricing['Days'], pricing['Price'])
result_price = mc_price.tukeyhsd()
print("Pairwise comparison for Price:")
print(result_price)

# For interaction terms, create a combined variable
pricing['interaction'] = pricing['Engagement'] + ' & ' + pricing['Price']
mc_interaction = MultiComparison(pricing['Days'], pricing['interaction'])
result_interaction = mc_interaction.tukeyhsd()
print("Pairwise comparison for Interaction:")
print(result_interaction)
```
#### Factorial Anova (Crossed factor design): Non parametric test 
```python
# Now we run the above test using a non-parametric approach 
# Note: Non parametric tests typically suffer from Type 1 error for interaction terms

# For non-parametric tests, we'll do pairwise Mann-Whitney U tests for each engagement level
p_values = []
test_pairs = []

for engagement in ['High', 'Low']:
    # Compare prices for each engagement level
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
        print(f"Mann-Whitney U test ({engagement}, {pair[0]} vs {pair[1]}): statistic = {u_stat:.4f}, p-value = {p_val:.4f}")

# Apply Holm-Bonferroni correction for multiple comparisons
reject, p_corrected, _, _ = multi.multipletests(p_values, method='holm')
print("\nCorrected p-values (Holm method):")
for i, pair in enumerate(test_pairs):
    print(f"{pair}: {p_corrected[i]:.4f}, reject H0: {reject[i]}")

```

### Generalized Linear Models 

Generalized Linear Models (GLM) extend Linear Models (LM) for studies with between factors to accommodate nominal (incl. binomial) or ordinal responses, or with non-normal response distributions (e.g., Poisson, exponential, gamma). All GLMs have a distribution and a link fn relating their factors to their response.
The GLM generalizes the LM, which is a GLM with a normal distribution and "identity" link fn.

Case study: Analyze user preference by Sex with multinomial logistic regression

Data: Table with user_id (Subject), Design preference (Pref) and Sex (M|F)

Questions to answer 
* Do users prefer one design over the other 
* Is the preference different by Gender 

```python
# Read data file
prefsABCsex = pd.read_csv("prefsABCsex.csv")
print(prefsABCsex.head())
prefsABCsex['Subject'] = prefsABCsex['Subject'].astype('category')
prefsABCsex['Sex'] = prefsABCsex['Sex'].astype('category')
print(prefsABCsex.describe())

# Plot preferences by gender
plt.figure(figsize=(10, 6))
pd.crosstab(prefsABCsex['Sex'], prefsABCsex['Pref']).plot(kind='bar')
plt.title('Design Preferences by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Create contingency table
contingency_table = pd.crosstab(prefsABCsex['Sex'], prefsABCsex['Pref'])
print("Contingency table:")
print(contingency_table)

# Chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square test: chi2 = {chi2:.4f}, p-value = {p:.4f}, dof = {dof}")

# Multinomial Logistic Regression
# Prepare data for MNLogit
# Create dummy variables for Sex (reference category will be 'F')
X = pd.get_dummies(prefsABCsex['Sex'], drop_first=True)
X = sm.add_constant(X)  # Add intercept

# Convert Pref to categorical 
y = pd.Categorical(prefsABCsex['Pref'])
y = pd.get_dummies(y)

# Fit the model
mnlogit_model = MNLogit(y, X)
mnlogit_results = mnlogit_model.fit()
print(mnlogit_results.summary())

# Anova test shows that the preferences are different.
# Next step is to do a pairwise comparison to understand the difference in preference for males 

# Binomial tests for males
male_counts = contingency_table.loc['M']
male_total = male_counts.sum()

print("\nBinomial tests for males:")
p_values_male = []
for pref in ['A', 'B', 'C']:
    count = male_counts[pref]
    p_val = stats.binom_test(count, male_total, p=1/3)
    p_values_male.append(p_val)
    print(f"Preference {pref}: count = {count}, p-value = {p_val:.4f}")

# Apply Holm-Bonferroni correction
reject, p_corrected, _, _ = multi.multipletests(p_values_male, method='holm')
print("\nCorrected p-values for males (Holm method):")
for i, pref in enumerate(['A', 'B', 'C']):
    print(f"Preference {pref}: {p_corrected[i]:.4f}, reject H0: {reject[i]}")

# Males preferred C over A and B

# Understanding the difference in preference for females
female_counts = contingency_table.loc['F']
female_total = female_counts.sum()

print("\nBinomial tests for females:")
p_values_female = []
for pref in ['A', 'B', 'C']:
    count = female_counts[pref]
    p_val = stats.binom_test(count, female_total, p=1/3)
    p_values_female.append(p_val)
    print(f"Preference {pref}: count = {count}, p-value = {p_val:.4f}")

# Apply Holm-Bonferroni correction
reject, p_corrected, _, _ = multi.multipletests(p_values_female, method='holm')
print("\nCorrected p-values for females (Holm method):")
for i, pref in enumerate(['A', 'B', 'C']):
    print(f"Preference {pref}: {p_corrected[i]:.4f}, reject H0: {reject[i]}")

# Females do not prefer A 

```

##### Using GLM for poisson distributed data 

__When the underlying data does not follow a normal distribution__

Previously, we had run Anova tests on Normally distributed data 
Using GLM, you can run Anova on data that is not normally distributed 

In the example below, we will be looking at error count data that follows a poisson distribution 

Case study: You have rolled 2 different versions of the user profile page and you want to see how many users have errors entering their details on the page. We will compare results of the 2 new versions with the existing version 

Poisson distribution is widely seen for count data. 

The Poisson distribution may be useful to model events such as
* The number of meteorites greater than 1 meter diameter that strike Earth in a year
* The number of patients arriving in an emergency room between 10 and 11 pm
* In a laser the number of photons hitting a detector in a particular time interval will follow Poisson distribution.

In our example, poisson is a candidate distribution because 
* Errors are discrete 
* We assume that occurrence of one error does not affect the probability that a second error will occur
* Users cannot make 2 errors at the same time 

```python
# Read data file
error_count = pd.read_csv("error_count.csv")
error_count['Design'] = pd.Categorical(error_count['Design'], 
                                      categories=['Orig', 'New1', 'New2'], 
                                      ordered=True)

# Descriptive statistics
print(error_count.groupby('Design')['Errors'].describe())
print(error_count.groupby('Design')['Errors'].agg(['mean', 'std']))

# Plot histograms
plt.figure(figsize=(15, 5))
for i, design in enumerate(['Orig', 'New1', 'New2']):
    plt.subplot(1, 3, i+1)
    plt.hist(error_count[error_count['Design'] == design]['Errors'], 
             bins=range(0, max(error_count['Errors'])+2))
    plt.title(f'Histogram of Errors for {design}')
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Design', y='Errors', data=error_count)
plt.title('Boxplot of Errors by Design')
plt.show()

# Test for Poisson distribution
from scipy.stats import poisson, chisquare

# Function to test goodness-of-fit for Poisson distribution
def test_poisson_fit(data):
    # Calculate lambda (mean) parameter
    lambda_param = data.mean()
    
    # Create observed frequency table
    observed = pd.Series(data).value_counts().sort_index()
    
    # Calculate expected frequencies based on Poisson PMF
    values = np.arange(0, observed.index.max() + 1)
    expected = poisson.pmf(values, lambda_param) * len(data)
    
    # For chi-square test, ensure expected counts are aligned with observed
    expected_counts = pd.Series(expected, index=values)
    observed_aligned = observed.reindex(expected_counts.index).fillna(0)
    
    # Combine categories with small expected counts (Chi-square requires expected counts >= 5)
    min_expected = 5
    if any(expected_counts < min_expected):
        # Find categories to combine
        mask_small = expected_counts < min_expected
        if sum(mask_small) > 0:
            last_valid_idx = expected_counts[~mask_small].index.max()
            
            # Combine small categories
            observed_grouped = observed_aligned.copy()
            expected_grouped = expected_counts.copy()
            
            observed_grouped.loc[last_valid_idx] = observed_aligned[observed_aligned.index >= last_valid_idx].sum()
            expected_grouped.loc[last_valid_idx] = expected_counts[expected_counts.index >= last_valid_idx].sum()
            
            # Drop combined categories
            observed_grouped = observed_grouped[observed_grouped.index <= last_valid_idx]
            expected_grouped = expected_grouped[expected_grouped.index <= last_valid_idx]
            
            # Update for test
            observed_aligned = observed_grouped
            expected_counts = expected_grouped
    
    # Run chi-square test
    chi2_stat, p_value = chisquare(observed_aligned, expected_counts)
    
    return chi2_stat, p_value, lambda_param

# Test each design for Poisson distribution
for design in ['Orig', 'New1', 'New2']:
    data = error_count[error_count['Design'] == design]['Errors']
    chi2_stat, p_value, lambda_param = test_poisson_fit(data)
    print(f"Poisson goodness-of-fit for {design}: lambda={lambda_param:.2f}, chi2={chi2_stat:.4f}, p-value={p_value:.4f}")

# Analyze using Poisson regression
import statsmodels.formula.api as smf

# Fit Poisson regression model
poisson_model = smf.glm(formula="Errors ~ Design", data=error_count, 
                        family=sm.families.Poisson()).fit()
print(poisson_model.summary())

# Check residuals
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(poisson_model.mu, poisson_model.resid_pearson)
plt.axhline(y=0, linestyle='--', color='gray')
plt.title('Pearson Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Pearson Residuals')

plt.subplot(1, 2, 2)
sm.qqplot(poisson_model.resid_deviance, line='s')
plt.title('QQ Plot of Deviance Residuals')
plt.tight_layout()
plt.show()

# Pairwise comparisons among levels of Design
# Create a DataFrame with predictions for each design
design_effects = pd.DataFrame({
    'Design': ['Orig', 'New1', 'New2'],
    'Effect': [poisson_model.params['Intercept'], 
              poisson_model.params['Intercept'] + poisson_model.params['Design[T.New1]'],
              poisson_model.params['Intercept'] + poisson_model.params['Design[T.New2]']]
})
print(design_effects)

# Multiple comparisons using Tukey's HSD
mc = MultiComparison(error_count['Errors'], error_count['Design'])
tukey_result = mc.tukeyhsd()
print(tukey_result)

# Calculate incidence rate ratios for easier interpretation
print("Incidence Rate Ratios (exp(coefficient)):")
irr = np.exp(poisson_model.params[1:])
print(irr)

print("Confidence Intervals for IRR:")
conf_int = np.exp(poisson_model.conf_int().iloc[1:])
print(conf_int)
```
