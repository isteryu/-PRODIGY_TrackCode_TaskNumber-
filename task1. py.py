import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

population_url = 'https://raw.githubusercontent.com/datasets/population/main/data/population.csv'

try:

    df = pd.read_csv(population_url)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have an active internet connection or replace the URL with a local file path.")
    
    df = pd.DataFrame(columns=['Country Name', 'Country Code', 'Year', 'Value'])



if not df.empty:
    df.rename(columns={'Value': 'Population'}, inplace=True)
    
    
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year].copy()
    
    df_latest['Population_M'] = df_latest['Population'] / 1_000_000
    
    df_countries = df_latest[df_latest['Country Code'].str.len() == 3].copy()
    
 
    print("\nMissing Values Check:")
    print(df_countries.isnull().sum())
    
    top_10_countries = df_countries.sort_values(by='Population_M', ascending=False).head(10)
    
    print(f"\nData prepared for the year: {latest_year}")




if not df_countries.empty:
    
    plt.figure(figsize=(14, 12))
    plt.suptitle(f'Population Distribution Analysis ({latest_year})', fontsize=18, weight='bold')


    plt.subplot(2, 1, 1)
    sns.barplot(x='Country Name', y='Population_M', data=top_10_countries, palette='viridis')
    plt.title('Top 10 Countries by Total Population (in Millions)')
    plt.xlabel('Country')
    plt.ylabel('Population (Millions)')
    plt.xticks(rotation=45, ha='right')
    # 

    
    plt.subplot(2, 1, 2)
  
    sns.histplot(df_countries['Population_M'], bins=50, kde=True, color='purple', log_scale=True)
    plt.title('Distribution of Population Across All Countries (Log Scale)')
    plt.xlabel('Population (Millions, Log Scale)')
    plt.ylabel('Number of Countries')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

else:
    print("\nCannot perform visualization: Data is empty.")
