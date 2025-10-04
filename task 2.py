import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


try:
    titanic_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(titanic_url)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have an active internet connection to load the file.")
    df = pd.DataFrame({
        'Survived': [0, 1, 1, 0], 'Pclass': [3, 1, 3, 2], 'Sex': ['male', 'female', 'female', 'male'], 
        'Age': [22.0, 38.0, 26.0, np.nan], 'SibSp': [1, 1, 0, 0], 'Parch': [0, 0, 0, 0], 
        'Fare': [7.25, 71.28, 7.92, 13.0], 'Embarked': ['S', 'C', 'S', 'S']
    })


print("\n--- Initial Data Inspection ---")
print(f"Shape of the dataset: {df.shape}")
print("\nData Types and Missing Values:")
df.info()



missing_values = df.isnull().sum()
print("\nMissing values before cleaning:")
print(missing_values[missing_values > 0].sort_values(ascending=False))


df.drop('Cabin', axis=1, inplace=True)


median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)


mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sort_values(ascending=False).head(3))



print("\n--- Summary Statistics (Numerical Features) ---")
print(df[['Age', 'Fare', 'Parch', 'SibSp']].describe().T)

survival_rate = df['Survived'].value_counts(normalize=True) * 100
print(f"\nSurvival Rate:\n{survival_rate}")

plt.figure(figsize=(12, 10))
plt.suptitle('Exploratory Data Analysis on Titanic Dataset', fontsize=18, weight='bold')

plt.subplot(2, 2, 1)
sns.barplot(x='Sex', y='Survived', data=df, palette='viridis')
plt.title('Survival Rate by Gender')
plt.ylabel('Proportion Survived')

plt.subplot(2, 2, 2)
sns.barplot(x='Pclass', y='Survived', data=df, palette='magma')
plt.title('Survival Rate by Passenger Class (Pclass)')
plt.ylabel('Proportion Survived')

plt.subplot(2, 2, 3)
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')

plt.subplot(2, 2, 4)

sns.histplot(df['Fare'], bins=50, kde=True, color='lightcoral')
plt.title('Fare Distribution (Skewed)')
plt.xlabel('Fare')
plt.ylabel('Count')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x='Survived', y='Age', data=df, palette=['lightcoral', 'lightgreen'])
plt.title('Age Distribution by Survival Status')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Age')
plt.show()

print("\n--- Key Insights ---")
print("1. **Gender:** Females had a significantly higher survival rate than males.")
print("2. **Passenger Class:** Passengers in Pclass 1 (First Class) had the highest survival proportion, highlighting the 'survival of the fittest' based on socio-economic status.")
print("3. **Age:** The age distribution of survivors vs. non-survivors is slightly different, with non-survivors having a higher median age. Children (lower age range) appear to have better survival chances (observed in the box plot).")