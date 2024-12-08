import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

diabetes_data = load_diabetes()
x = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
y = pd.Series(diabetes_data.target)


st.title("Diabetes Prediction Analysis with Linear Regression")


st.header("Overview")
st.write("""
The purpose of this analysis is to forecast the progress of diabetes using a number of health indicators. Ten features pertaining to patient demographics and health information are included in the diabetes dataset from Sklearn, 
which we are implementing. To model how these factors relate to the target variable, we will employ linear regression.
""")

st.subheader("Dataset Structure")
st.write("""
The dataset contains the following columns:
- age: Age of the patient
- sex: Gender of the patient
- bmi: Body mass index
- bp: Average blood pressure
- s1: Total serum cholesterol (tc)
- s2: Low-density lipoproteins (ldl)
- s3: High-density lipoproteins (hdl)
- s4: Total cholesterol / HDL (tch)
- s5: Possibly log of serum triglycerides level (ltg)
- s6: Blood sugar level (glu)
""")


st.subheader("Basic Descriptive Statistics")
st.write(x.describe())


missing_values = x.isnull().sum()
missing_values_df = missing_values.reset_index()
missing_values_df.columns = ['Feature', 'Missing Count']
st.subheader("Missing Values")
st.write(missing_values_df)

st.subheader("Feature Distributions")
fig, axes = plt.subplots(5, 2, figsize=(14, 12))
axes = axes.ravel()


for i, feature in enumerate(x.columns):
    sns.histplot(x[feature], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {feature}", fontsize=12)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")


plt.tight_layout()
st.pyplot(fig)


st.subheader("Correlation Heatmap")
correlation = x.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Features")
st.pyplot(plt)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1412)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R² Score: {r2}")

st.subheader("Predicted vs Actual Values")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Predicted vs True Values")
st.pyplot(plt)

st.subheader("Interactive Analysis")

from sklearn.linear_model import LinearRegression

selected_feature = st.selectbox("Feature to check against Target", x.columns)

X = x[[selected_feature]]
y = y

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(x[selected_feature], y, color='blue', alpha=0.7, label="Actual Data")
plt.plot(x[selected_feature], y_pred, color='red', linewidth=2, label="Regression Line")
plt.xlabel(selected_feature)
plt.ylabel("Target")
plt.title(f"{selected_feature} vs Target")
st.pyplot(plt)

st.subheader("Conclusions")
st.write("""
From the analysis and model evaluation, we can conclude the following:

1. **Model Performance:**
   - The linear regression model's performance is evaluated using the Mean Squared Error (MSE) and R² score. The MSE quantifies the average squared difference between predicted and actual values, while the R² score indicates the proportion of variance explained by the model. A higher R² score means a better fit of the model to the data.
   
2. **Feature Correlations:**
   - The correlation heatmap shows the relationships between different features. Strong correlations between certain features (e.g., s1 and s2) might indicate that some variables could be more predictive of diabetes progression than others.
   
3. **Prediction Insights:**
   - The scatter plot comparing predicted vs actual values shows that while the model generally performs well, there are still some deviations. The regression line provides insight into the general trend, indicating that the model captures the overall pattern but might not be perfect for every prediction.

4. **Interactive Analysis:**
   - The interactive feature allows users to explore the relationship between individual features (such as BMI or age) and the target variable. This can help understand which features most influence diabetes progression and give a deeper insight into the data.

In conclusion, this analysis serves as an initial exploration of the diabetes progression dataset, highlighting key patterns and providing insights into the relationships between the features and the target variable.
""")
