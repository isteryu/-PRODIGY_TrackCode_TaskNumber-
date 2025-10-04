import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style('darkgrid')

# --- 1. Data Loading ---
# This dataset is commonly used for Prodigy's Task 1, often sourced from the World Bank.
# We will load a simplified version directly from a public repository.
population_url = 'https://raw.githubusercontent.com/datasets/population/main/data/population.csv'

try:
    # Read the dataset
    df = pd.read_csv(population_url)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have an active internet connection or replace the URL with a local file path.")
    # Fallback to an empty DataFrame if loading fails
    df = pd.DataFrame(columns=['Country Name', 'Country Code', 'Year', 'Value'])


# --- 2. Data Cleaning and Preparation ---

if not df.empty:
    # Rename 'Value' to 'Population' for clarity
    df.rename(columns={'Value': 'Population'}, inplace=True)
    
    # Filter for the most recent year available in the dataset
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year].copy()
    
    # Convert population to millions for better visualization scale
    df_latest['Population_M'] = df_latest['Population'] / 1_000_000
    
    # Filter out aggregated regional data (rows where 'Country Code' is not 3 letters)
    # This step is crucial for visualizing *country* distributions only.
    df_countries = df_latest[df_latest['Country Code'].str.len() == 3].copy()
    
    # Check for missing values (usually minimal in this cleaned dataset)
    print("\nMissing Values Check:")
    print(df_countries.isnull().sum())
    
    # Sort for the top countries
    top_10_countries = df_countries.sort_values(by='Population_M', ascending=False).head(10)
    
    print(f"\nData prepared for the year: {latest_year}")


# --- 3. Visualization ---

if not df_countries.empty:
    
    plt.figure(figsize=(14, 12))
    plt.suptitle(f'Population Distribution Analysis ({latest_year})', fontsize=18, weight='bold')

    # A. Bar Chart: Top 10 Countries by Population (Categorical Variable Distribution)
    plt.subplot(2, 1, 1)
    sns.barplot(x='Country Name', y='Population_M', data=top_10_countries, palette='viridis')
    plt.title('Top 10 Countries by Total Population (in Millions)')
    plt.xlabel('Country')
    plt.ylabel('Population (Millions)')
    plt.xticks(rotation=45, ha='right')
    # 

    # B. Histogram: Distribution of Country Populations (Continuous Variable Distribution)
    plt.subplot(2, 1, 2)
    # Use a log scale on the x-axis to better visualize the highly right-skewed data
    sns.histplot(df_countries['Population_M'], bins=50, kde=True, color='purple', log_scale=True)
    plt.title('Distribution of Population Across All Countries (Log Scale)')
    plt.xlabel('Population (Millions, Log Scale)')
    plt.ylabel('Number of Countries')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

else:
    print("\nCannot perform visualization: Data is empty.")
