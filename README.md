# Bank_Customer_Churn_Analysis


```python
df.info()
```

![info](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/column%20info%20about%20dataset.png?raw=true)

```python
# Print the column names of the dataset.
print(df.columns) 
```

![column_names](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/column_names.png?raw=true)

```python
# Standardize the column names: convert to lowercase and replace spaces with underscores.

# Replacing all spaces with underscore(_) and converting the entire string to lowercase

df.columns = df.columns.str.lower().str.replace(" ", "_")

# Showing the column headers again
df.columns
```

![column_names](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/columns_names_with_underscore_correction.png?raw=true)

### Handling Missing Values
```python
# Check for missing values
print(df.isnull().sum()) 

# What it does:
# df.isnull() → Returns a DataFrame of the same shape as df, with True where a value is missing (NaN) and False otherwise.
# .sum() → Counts the number of True values (i.e., missing values) in each column.
```

![null_values](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/check_null_before_cleaning.png?raw=true)

```python
# Impute missing 'salary' values with median
df['salary'].fillna(df['salary'].median(), inplace=True)

# Impute missing 'balance' values with 0
df['balance'].fillna(0, inplace=True)

# Impute missing 'satisfaction_score' with median
df['satisfaction_score'].fillna(df['satisfaction_score'].median(), inplace=True)

# Drop rows where 'gender' is missing
df = df.dropna(subset=['gender'])

# Fill missing 'card_type' with the most frequent value (mode)
df['card_type'].fillna(df['card_type'].mode()[0], inplace=True)
```

```python
# Print the count of missing values in each column
print(df.isnull().sum())  # Count of NaN values
```

![after](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/check_null_after_cleaning.png?raw=true)

```python
#  Convert 'card_type' to uppercase and Strip whitespace from 'card_type'
df["card_type"] = df["card_type"].str.upper().str.strip()
df['card_type'].unique()
```

![card_type](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/upper_case_all_card_type.png?raw=true)

### Explore Numeric Values 

```python
# Identify all numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
print(numeric_columns) 
```

![numeric](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/numeric_columns.png?raw=true)

```python
# Count unique values in each numeric column
unique_counts = {col: df[col].nunique() for col in numeric_columns}

print(unique_counts)
```

![count](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/numeric_columns_count.png?raw=true)

### Outlier Detection and Treatment

```python
df['salary'].plot(kind='box',vert=False)
plt.title("Salary Distribution")
plt.show()
```

![outlier](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/salary_distribution_with_outliers.png?raw=true)

```python
# Calculate IQR
Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

print(f"lower bound :- {lower_bound}\n")
print(f"upper bound :- {upper_bound}\n")

# Identify outliers
salary_outliers = df[(df["salary"] < lower_bound) | (df["salary"] > upper_bound)]

print("Outlier Customers Based on Salary:")
print(salary_outliers[["salary"]])

print()

outlier_count = ((df["salary"] < lower_bound) | (df["salary"] > upper_bound)).sum()
print("Count of Outlier Customers Based on Salary:", outlier_count)
```
![upper](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/salary_upper_lower_bound.png?raw=true)

```python
# removing the upper bound outliers
df["salary"] = np.where(df["salary"] > upper_cap, upper_cap, df["salary"])

df['salary'].plot(kind='box',vert=False)
plt.show()
```

![outlier](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/salary_distribution_without_outliers.png?raw=true)

```python
df['balance'].plot(kind='box',vert=False)
plt.title("Balance Distribution")
plt.show()
```

![balance](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/balance_distribution_with_outliers.png?raw=true)

```python
# Step 1: Calculate IQR
import numpy as np
Q1 = df["balance"].quantile(0.25)
Q3 = df["balance"].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Define lower and upper caps
lower_cap = Q1 - 1.5 * IQR
upper_cap = Q3 + 1.5 * IQR

print(f"lower_bound : {lower_cap}")
print(f"upper_bound : {upper_cap}")

# Step 3: Modify the original column directly
df["balance"] = np.where(df["balance"] > upper_cap, upper_cap, df["balance"])
```

![values](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/balance_upper_lower_bound.png?raw=true)

```python
df['balance'].plot(kind='box',vert=False)
plt.show()
```

![values](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/balance_without_outliers.png?raw=true)

### EDA

```python
# Compute the mean and median for key numeric columns: Salary , Balance and Credit Score

mean_salary = round(df["salary"].mean())
median_salary = round(df["salary"].median())

mean_balance = round(df["balance"].mean())
median_balance = round(df["balance"].median())

mean_credit_score = round(df["credit_score"].mean())
median_credit_score = round(df["credit_score"].median())

print(f"Mean Salary: {mean_salary}, Median Salary: {median_salary}\n")
print(f"Mean Balance: {mean_balance}, Median Balance: {median_balance}\n")
print(f"Mean Credit Score: {mean_credit_score}, Median Credit Score: {median_credit_score}\n")
```

![mean_median](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/mean_median_mode.png?raw=true)

```python
# Count and display how many customers fall into each category of: Gender, Card Type, HasLoan, HasFD

gender_count = df["gender"].value_counts()
print("Gender Distribution:\n", gender_count)

# Plot gender distribution
plt.figure(figsize=(6, 4))
gender_count.plot(kind='bar')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

![values](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/gender_distribution_values.png?raw=true)

![visual](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/gender_distribution_chart.png?raw=true)

```python
card_type_count = df["card_type"].value_counts()
print("\nCard Type Distribution:\n", card_type_count)

# Plot card type distribution
plt.figure(figsize=(6, 4))
card_type_count.plot(kind='bar')
plt.title("Card Type Distribution")
plt.xlabel("Card Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

![card_type](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/card_type_distribution_values.png?raw=true)

![visual](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/card_type_distribution_visual.png?raw=true)

```python
loan_status_count = df["hasloan"].value_counts()
print("\nLoan Status Distribution:\n", loan_status_count)

# Plot loan status distribution
plt.figure(figsize=(6, 4))
loan_status_count.plot(kind='bar')
plt.title("Loan Status Distribution")
plt.xlabel("Has Loan")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```
![val](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/loan_status_values.png?raw=true)

![has_loan](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/loan_status_distribution_visual.png?raw=true)

```python
fd_status_count = df["hasfd"].value_counts()
print("\nFixed Deposit Status Distribution:\n", fd_status_count)

# Plot fixed deposit status distribution
plt.figure(figsize=(6, 4))
fd_status_count.plot(kind='bar')
plt.title("Fixed Deposit Status Distribution")
plt.xlabel("Has FD")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```
![vis](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/fixed_deposit_values.png?raw=true)

![has_fd](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/fixed_deposit_distribution_visuals.png?raw=true)

```python
# Select a random sample of 200 customers from the dataset.
# Plot scatter plots to explore how two numeric variables relate to each other, plot for Credit Score vs Balance

# Subset of data (random sampling) around 200 datapoints are taken here
df_sample = df.sample(n=200, random_state=42)

# Scatter plot to detect anomalies in Credit Score vs Balance
plt.figure(figsize=(10,5))
sns.scatterplot(x=df_sample["credit_score"], y=df_sample["balance"])
plt.title("Credit Score vs Balance")
plt.xlabel("Credit Score")
plt.ylabel("Balance")
plt.show()
```

![sample](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/credit_score_vs_balance_scatter_plot.png?raw=true)

### Feature Engineering

```python
# Add the following new columns to the dataset:
# Debt-to-Income Ratio
#  Formula: (Balance + (HasLoan × Salary × 0.3)) / Salary

# 1. Financial Stability Indicator
df["Debt-to-Income Ratio"] = (df["balance"] + (df["hasloan"] * df["salary"] * 0.3)) / df["salary"]

print(df[["first_name", "Debt-to-Income Ratio"]].tail())
```

![debt](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/debt_to_income_ratio.png?raw=true)

```python
# Create new column calculating Loyalty Score
# Formula: (Tenure × Satisfaction Score) / (1 + Count of Complains)

df["Loyalty Score"] = (df["tenure"] * df["satisfaction_score"]) / (1 + df["count_of_complains"])

# Display dataset with new features
print(df[["first_name","Loyalty Score"]].head())
```

![loyalty](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/loyalty_score.png?raw=true)

### analyze complaints by state 

```python
# Compute the average number of complaints per state using transform().

df["State Avg Complaints"] = df.groupby("state")["count_of_complains"].transform("mean")
```

```python
# For each customer, compare their complaint count to their state’s average.

# Create a flag called High Complainer:
# 1 if their complaints are above the state average
# 0 otherwise
df["Above State Avg Complaints"] = df["count_of_complains"] > df["State Avg Complaints"]

df["High Complainer"] = df["Above State Avg Complaints"].astype(int)
```

```python
df[['first_name','state','count_of_complains','High Complainer']].head(10)
```

![visual](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/complaints_higher_than_the_avg_state_complaints.png?raw=true)

### Univariate Analysis - Categorize Customers by Salary 

```python
# Create a new column called Salary Category with the following buckets:
# Low (≤ 50,000)
# Medium (50,001 – 100,000)
# High (100,001 – 150,000)
# Very High (150,001 – 200,000)
# Above 2 Lakhs (> 200,000)
# Count how many customers fall into each group and plot the result.


df["Salary Category"] = np.where(df["salary"] <= 50000, "Low",
                         np.where(df["salary"] <= 100000, "Medium",
                            np.where(df["salary"] <= 150000, "High",
                                np.where(df["salary"] <= 200000, "Very High", "Above 2 Lakhs"))))

# Count customers in each category
salary_counts = df["Salary Category"].value_counts()

# series datatype
print(salary_counts)

# Plot salary distribution
# Here, index indicates the labels and values indicate total no of customers that fall into that label
plt.bar(salary_counts.index, salary_counts.values, color="skyblue")
plt.xlabel("Salary Category")
plt.ylabel("Number of Customers")
plt.title("Customer Distribution by Salary Category")
plt.show()

```

![income](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/churn_rate_by_income_group_values.png?raw=true)

![income](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/churn_rate_by_income_group_visual.png?raw=true)

### compare customer segement using grouped statistics
### Bivariate analysis 

```python
# Calculate Average Number of Products Based on Customer Tenure

tenure_product_analysis = df.groupby("tenure")["num_of_products"].mean()
# Display results
print("Average Number of Products Based on Customer Tenure:")
print(tenure_product_analysis)
```

![avg]()

### Multivariate analysis 

```python
# Grouping by churn status to analyze salary and product usage
churn_analysis = df.groupby("exited")[["salary", "num_of_products"]].mean()
print(churn_analysis)
```

```python
# Bar chart to compare salary for exited vs. retained customers
plt.figure(figsize=(6, 4))
plt.bar(["stayed", "exited"], churn_analysis["salary"], color=["green", "red"])
plt.title("Average Salary - Stayed vs. Exited")
plt.ylabel("Average Salary")
plt.show()
```

```python
# Bar chart to compare product usage for exited vs. retained customers
plt.figure(figsize=(6, 4))
plt.bar(["stayed", "exited"], churn_analysis["num_of_products"], color=["green", "red"])
plt.title("Average Number of Products - Stayed vs. Exited")
plt.ylabel("Number of Products")
plt.show()
```

### Demographic Analysis 

### visualize customers by churn status 

```python
# Count no of churned and non churned customers 
churn_counts = df['exited'].value_counts()

print(churn_counts)

# Label retained and churned 
labels = ['Retained', 'Churned']

plt.figure(figsize=(6, 6))
churn_counts.plot(kind='pie', labels=labels, autopct='%1.1f%%')
plt.title('Customer Churn Proportion')
plt.show()
```
![val](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/customer_churn_values.png?raw=true)

![values](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/customer_churn_pie_chart.png?raw=true)

### churn variation across states 

```python
region_churn_counts = df.groupby('state')['exited'].sum()

print(region_churn_counts.sort_values(ascending=False))

plt.figure(figsize=(8, 6))
region_churn_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Churn Proportion by State')
plt.show()
```

![churn](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/churn_by_state_values.png?raw=true)

![vis](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/churn_by_state_pie_chart.png?raw=true)

### visualize age vs exit status 

```python

# Count churned and non-churned customers by age
churned_count = df[df['exited'] == 1].groupby('age').size()
non_churned_count = df[df['exited'] == 0].groupby('age').size()

# print(f'churned count :- {churned_count}\n')
# print(f'non churned count :-{non_churned_count}\n)

# Plot in a scatter plot
sns.scatterplot(x=churned_count.index, y=churned_count.values, color='red', label='Churned')
sns.scatterplot(x=non_churned_count.index, y=non_churned_count.values, color='green', label='Non-Churned')

plt.title("Scatter Plot: Age vs Number of Customers (Churned vs Non-Churned)")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.legend()
plt.show()
```

![age](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/age_vs_num_of_products_scatter_plot.png?raw=true)

### visualize age distribution and exit status 

```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='exited', y='age', data=df)
plt.title("Box Plot: Age Distribution by Churn Status")
plt.xlabel("Churn Status (Exited)")
plt.ylabel("Age")
plt.show()
```

![visual](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/box_plot_age_distribution_by_churn_status.png?raw=true)

### visualize gender and exit status 

```python
# Filter for customers who have exited (churned)
churned_customers_df = df[df['exited'] == 1]

# Count total customers per gender
total_customers_by_gender = df.groupby('gender')['exited'].count()

# Count churned customers per gender
churned_customers_by_gender = churned_customers_df.groupby('gender')['exited'].count()
print(churned_customers_by_gender)

# Step 4: Calculate churn rate manually
churn_rate_by_gender = (churned_customers_by_gender / total_customers_by_gender) * 100

print(churn_rate_by_gender)
# Step 5: Plot churn rate by gender
plt.figure(figsize=(10, 6))
sns.barplot(x=churn_rate_by_gender.index, y=churn_rate_by_gender.values)
plt.title("Churn Rate Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Churn Rate (%)")
plt.show()

```

![churn](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/churn_rate_by_gender_values.png?raw=true)

![vis](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/chuen_rate_by_gender_visual.png?raw=true)

### churn rate by region and gender 

```python
# Step 1: Group data by state and gender
grouped_data = df.groupby(['state', 'gender'])

# Step 2: Count total customers for each group
total_customers = grouped_data['exited'].count()

# Step 3: Count churned customers for each group
churned_customers = grouped_data['exited'].sum()  # Since 'exited' is 1 for churned customers

# Step 4: Calculate churn rate manually
churn_rate = (churned_customers / total_customers) * 100

churn_rate_unstacked = churn_rate.unstack()

print(churn_rate_unstacked)

# Step 5: Plot a bar chart 
plt.figure(figsize=(12, 6))

churn_rate_unstacked.plot(kind='bar', edgecolor='black')

# Step 6: Customize the plot
plt.title('Churn Rate by State and Gender')
plt.xlabel('State')
plt.ylabel('Churn Rate (%)')
plt.show()

```

![churn](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/churn_Rate_by_state_and_gender_values.png?raw=true)

![vis](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/churn_rate_by_state_and_gender_visual.png?raw=true)

### relationship between income groups - salary and churn rate

```python
# Step 1: Create income bins (ranges)
bins = [0, 30000, 50000, 70000, 100000, 150000]  # Example income ranges
labels = ['<30K', '30K-50K', '50K-70K', '70K-100K', '>100K']

# Step 2: Assign income group labels
df['income_group'] = pd.cut(df['salary'], bins=bins, labels=labels)

# Step 3: Count total customers in each income group
total_customers_by_income = df.groupby('income_group')['exited'].count()

# Step 4: Count churned customers in each income group
churned_customers_by_income = df.groupby('income_group')['exited'].sum()

# Step 5: Calculate churn rate manually and convert to percentage
churn_rate_by_income = (churned_customers_by_income / total_customers_by_income) * 100

print(churn_rate_by_income)

# Step 6: Plot a bar chart
plt.figure(figsize=(10, 6))
churn_rate_by_income.plot(kind='bar')
plt.title('Churn Rate by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Churn Rate (%)')
plt.show()
```

![income](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/churn_rate_by_income_group_values.png?raw=true)

![vis](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/churn_rate_by_income_group_visual.png?raw=true)

```python
# Step 1: Define bins and labels for age groups
age_bins = [18, 30, 45, 60, 100]  # Age ranges
age_labels = ['18-30', '31-45', '46-60', '60+']  # Corresponding labels

# Step 2: Apply pd.cut() to create age groups
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

print(df.head())
```

### churn rate across region and gender 

```python
# Calculate the churn rate by region and gender

# Step 1: Group data by region and gender
grouped_data = df.groupby(['region', 'gender'])

# Step 2: Count total customers for each region-gender group
total_customers = grouped_data['exited'].count()

# Step 3: Count churned customers for each region-gender group
churned_customers = grouped_data['exited'].sum()  # 'exited' is 1 for churned customers

# Step 4: Calculate churn rate manually
churn_rate = (churned_customers / total_customers) * 100

# Step 5: Convert the series into a DataFrame for visualization
churn_rate_region_gender = churn_rate.reset_index(name='churn_rate')

print(churn_rate_region_gender)

# Step 6: Plot a bar chart to visualize churn rate by region and gender
plt.figure(figsize=(12, 6))
sns.barplot(x='region', y='churn_rate', hue='gender', data=churn_rate_region_gender)

# Step 7: Customize the plot
plt.title('Churn Rate by Region and Gender')
plt.xlabel('Region')
plt.ylabel('Churn Rate (%)')

# Step 8: Display the plot
plt.show()

```

![val](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/churn_rate_by_region_gender.png?raw=true)

![vis](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/churn_rate_by_region_gender_visual.png?raw=true)

### average satisfaction for age groups 

```python
# Step 1: Filter for churned customers where 'exited' == 1
churned_customers = df[df['exited'] == 1]

# Group by 'age_group' and calculate the average satisfaction score for churned customers
# Step 2: Count total churned customers per age group
total_churned_per_age_group = churned_customers.groupby('age_group')['satisfaction_score'].count()

# Step 3: Calculate the mean of satisfaction scores per age group
avg_satisfaction_churned = churned_customers.groupby('age_group')['satisfaction_score'].mean().reset_index(name="avg_satisfaction_score")

avg_satisfaction_churned = avg_satisfaction_churned.dropna(subset=['avg_satisfaction_score'])

print(avg_satisfaction_churned)

plt.figure(figsize=(12, 6))
sns.barplot(x='age_group', y='avg_satisfaction_score', data=avg_satisfaction_churned)
plt.title('Churned Customer Satisfaction by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Satisfaction Score')
plt.show()

```

![age group](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/age_group_avg_satisfaction_values.png?raw=true)

![vis](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/age_group_avg_satisfaction_visuals.png?raw=true)

### count of customers by employment type highlghting churned vs non-churned customers 

```python
# Get numeric counts for each employment type and churn status
counts = df.groupby(['employment_type', 'exited']).size().reset_index()
# print(counts)
# Count Plot: Employment Type vs. Churn
plt.figure(figsize=(10, 6))
sns.countplot(x='employment_type', hue='exited', data=df, palette='coolwarm')
plt.title("Employment Type vs Churn")
plt.xlabel("Employment Type")
plt.ylabel("Number of customers")
plt.show()
```

![cus](https://github.com/parthpatoliya97/Bank_Customer_Churn_Analysis/blob/main/images/employment_type_vs_churn.png?raw=true)

### Product analysis

### credit card distribution among churned customers 

```python
# Filter the dataset where 'isexited' == 1 (indicating customers who have exited)
exited_customers = df[df['exited'] == 1]

# # Create a count plot for credit card ownership among exited customers
plt.figure(figsize=(6, 4))
sns.countplot(x=exited_customers["hascrcard"],palette="coolwarm")
plt.title("Credit Card Ownership Distribution Among Exited Customers")
plt.xlabel("Has Credit Card (0 = No, 1 = Yes)")
plt.ylabel("Number of churned Customers")
plt.show()

```

![vis]()

```python
# Step 1: Group by 'hascrcard' and 'exited' and count number of customers in each group
group_counts = df.groupby(['hascrcard', 'exited']).size().reset_index(name='count')

print(group_counts)

# Step 2: Calculate total number of customers
total_customers = group_counts['count'].sum()
#
print(total_customers)

# Step 3: Calculate percentage as (group count / total count) * 100
group_counts['percentage'] = (group_counts['count'] / total_customers) * 100

# Step 4: Plot using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(data=group_counts, x='hascrcard', y='percentage', hue='exited', palette="coolwarm", errorbar=None)

# Enhancing the plot
plt.title('Global Customer Churn Percentage by Credit Card Status')
plt.xlabel('Has Credit Card (0 = No, 1 = Yes)')
plt.ylabel('Percentage of Total Customers')
plt.legend(title='Churned (Exited)')
plt.tight_layout()
plt.show()
```

### credit card type distribution across churn status 

```python
# Step 1: Group by 'card_type' and 'exited' and count number of customers in each group
card_counts = df.groupby(['card_type', 'exited']).size().reset_index(name='count')

# Step 2: Calculate total number of customers
total_customers = card_counts['count'].sum()

# Step 3: Calculate percentage as (group count / total count) * 100 and round
card_counts['percentage'] = ((card_counts['count'] / total_customers) * 100).round(2)

print(card_counts)

# Step 4: Plot using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(data=card_counts, x='card_type', y='percentage', hue='exited', palette="coolwarm", errorbar=None)

# Enhancing the plot
plt.title('Credit card type percentage across churn status')
plt.xlabel('Credit card type')
plt.ylabel('Percentage of Total Customers')
plt.tight_layout()
plt.show()

```


### loan ownership vs churn 

```python
plt.figure(figsize=(6, 4))
sns.countplot(x=df["hasloan"], hue=df["exited"], palette="coolwarm")
plt.title("HasLoan Ownership Distribution by Churned Status")
plt.xlabel("HasLoan (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()
```

### number of products vs churn 

```python
# Filter the dataset where 'isexited' == 1 (indicating customers who have exited)
exited_customers = df[df['exited'] == 1]

# Credit Card Type Distribution
# plt.figure(figsize=(8, 4))
sns.countplot(x=exited_customers["card_type"], palette="coolwarm")
plt.title("Credit Card Type Distribution")
plt.xlabel("Card type")
plt.ylabel("No of churned customers")
plt.show()
```

### product distribution across churn 

```python
# Step 1: Group by churn status and each product
crcard = df[df['hascrcard'] == 1].groupby('exited').size()
print("Credit card: ")
print(crcard)

loan = df[df['hasloan'] == 1].groupby('exited').size()
print("Loan count: ")
print(loan)

fd = df[df['hasfd'] == 1].groupby('exited').size()
print("HasLoan count: ")
print(fd)

# Step 2: Combine into a DataFrame
product_dist = pd.DataFrame({
    'Credit Card': crcard,
    'Loan': loan,
    'Fixed Deposit': fd
}) 

# Step 1: Plot the bar chart
ax = product_dist.plot(kind='bar', figsize=(5, 6))

# Step 2: Automatically label bars
ax.bar_label(ax.containers[0], fontsize=9, padding=3)
ax.bar_label(ax.containers[1], fontsize=9, padding=3)
ax.bar_label(ax.containers[2], fontsize=9, padding=3)


plt.title('Product Distribution Across Churn Status')
plt.xlabel('Churn status')
plt.ylabel('Number of Customers')
plt.legend(title='Products')
plt.tight_layout()
plt.show()

```

### average product usage by tenure groups across churn status 

```python
# Step 1: Create tenure groups ---> Make this step simple by using array of value instead
tenure_bins = list(range(0, 41, 5))
tenure_labels = [f'{i}<={i+5}' for i in range(0, 36, 5)]
df['tenure_group'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels, right=False)

# Step 2: Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df,
    x='tenure_group',
    y='numofproducts',
    hue='exited',
    palette='muted',
    errorbar=None
)
plt.xlabel('Tenure Group (Years)')
plt.ylabel('Average Number of Products')
plt.title('Tenure Group vs Avg Number of Products by Churn Status')
plt.legend(title='Exited', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

```

### average number of products usage by given credit score ranges across churned status 

```python
# Step 1: Create credit score bins

credit_bins = [200, 400, 600, 800, 1000, 1200]
credit_labels = ['200-399', '400-599', '600-799', '800-999', '1000-1199']

df['credit_score_range'] = pd.cut(df['creditscore'], bins=credit_bins, labels=credit_labels)

# Step 2: Set figure size
# plt.figure(figsize=(10, 6))

# Step 3: Create a bar plot for credit score range vs. number of products across churned status
ax = sns.barplot(
    data=df,
    x='credit_score_range',
    y='numofproducts',
    hue='exited',
    palette='coolwarm',
    errorbar=None
)

# Step 4: Annotate each bar with average value
for p in ax.patches:
    height = p.get_height()
    if pd.notnull(height) and height > 0:
        ax.text(
            x=p.get_x() + p.get_width() / 2,
            y=height + 0.02,
            s=f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )


# Step 4: Label the axes and title
plt.title('Average Product Usage by Credit Score Range and Churn Status')
plt.xlabel('Credit Score Range')
plt.ylabel('Average Number of Products')
plt.legend(title='Churned (1 = Yes, 0 = No)')
plt.tight_layout()
plt.show()

```

### Feedback Analysis 
### 🧠 Assumption 1:
Customers with low satisfaction and low loyalty points are more likely to churn.

✅ Approach:
- You're plotting two continuous variables: satisfaction_score and point_earned.
- A scatter plot helps detect relationships, clusters, or patterns between these two variables.
- Coloring by churn_status adds a third visual dimension — allowing us to visually detect trends in churn behavior based on combinations of satisfaction and rewards.



```python
# Map 'exited' column to categorical labels
df["churn_status"] = df["exited"].map({0: "Stayed", 1: "Left"})

colors = {0: 'green', 1: 'red'}

sns.scatterplot(
    data=df,
    x='satisfaction_score',
    y='point_earned',
    hue='exited',
    palette=colors # Apply the custom palette here
)

plt.title("Satisfaction vs Points Earned by Churn")
plt.xlabel("Satisfaction Score")
plt.ylabel("Points Earned")
plt.legend(title="Churn Status", labels=["Stayed", "Churned"]) # Ensure labels match the palette order
plt.show()
```


### 🧠 Assumption 2:
Customers who churn tend to have lower satisfaction scores than those who stay.


✅ Approach:
- Compare the distribution of satisfaction_score for churned vs non-churned customers.
- Plot both on the same axis for visual comparison.

📊 Why this chart?
- A histogram shows how a variable is spread out.
- When overlaid, we can spot shifts in the score distribution between groups.

```python
satisfied_stayed = df[df['exited'] == 0]['satisfaction_score']
satisfied_churned = df[df['exited'] == 1]['satisfaction_score']

# Stacked Histogram
plt.hist(
    [satisfied_stayed, satisfied_churned],
    bins=10,
    stacked=True,
    color=['green', 'red'],
    label=['Stayed', 'Churned'],
    edgecolor='black'
)

# Labels & title
plt.title("Satisfaction Score Distribution by Churn")
plt.xlabel("Satisfaction Score")
plt.ylabel("Frequency")
plt.legend(title="Churn Status")

plt.show()

```


### 🧠 Assumption 3:
Among those who complained, a higher proportion ended up churning.

✅ Approach:
- Focus only on rows where complain = 1
- Show what portion of them stayed vs left

📊 Why this chart?
- A pie chart visually highlights proportions within a group — perfect for categorical splits like churn within complainants.

```python
complainers = df[df['complain'] == 1]

churn_distribution = complainers['exited'].value_counts(normalize=True) * 100

print("Churn Distribution Among Complainers:\n")
print(churn_distribution)

plt.pie(
    churn_distribution,
    labels=['Stayed (0)', 'Churned (1)'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['green', 'red']
)

plt.title("Churn Distribution Among Complainers")
plt.show()
```

### Assumption 4
Customers with high salary and high balance are less likely to churn
Approach:
- You’re dealing with two continuous numerical variables: salary and balance.
- A scatter plot is ideal to visualize how those two interact and to detect patterns or clusters related to churn.
- Coloring by churn_status adds an extra dimension (a third variable) visually, making it easier to spot correlations between financial metrics and churn behavior.

```python
sample_df = df.sample(500, random_state=42)

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=sample_df,
    x='salary',
    y='balance',
    hue='exited',
    alpha=0.6
)

plt.title("Salary vs Balance by Churn Status")
plt.xlabel("Salary")
plt.ylabel("Balance")
plt.legend(title="Churn Status", labels=["Stayed", "Churned"])
plt.show()

```


### Assumption 5
- Churn rate differs between FD holders and non-holders

Approach:
- Filter the dataset to include only FD holders (hasfd == 1).
- Use value_counts(normalize=True) gives percentage proportions of churned vs. non-churned customers among FD holders.
- Plot pie chart to show part-to-whole relationships.

```python
fd_holders=df[df['hasfd']==1]

churn_dist = fd_holders['churn_status'].value_counts(normalize=True).round(2)

plt.figure(figsize=(7,7))
plt.pie(
    churn_dist.values,
    labels=churn_dist.index,
    autopct='%1.2f%%',
    startangle=90
)

plt.title("Churn Distribution Among FD Holders")
plt.tight_layout()
plt.show()
```

