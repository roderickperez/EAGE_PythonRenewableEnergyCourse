# SQL + Pandas

In this section we will cover a step-by-step example to integrate the energy production data stored in a database (SQLite) and then using Python to analyze and plot the data. This guide walks through each step, from loading CSV data into an SQLite database to querying and plotting the energy production data for different countries.

This sections includes downloading the data from individual `.csv` files, loading them into a SQLite database, and visualizing the energy production using Python.

This guide shows how to:
1. Download CSV files from Eurostat.
2. Load data into a SQLite database.
3. Use Python to query and visualize the data.

## Step 1: Download the Data from Eurostat

The datasets are available on the Eurostat website under [Energy Statistics](https://ec.europa.eu/eurostat/web/energy). Let’s assume the data files have been downloaded and stored in your `~/Downloads` folder as `.csv` files.

For this example, we are using the following datasets:
- coal: `../data/section4/euroStat/nrg_cb_pem_page_spreadsheet_coal.xlsx`
- nonRenewable: `../data/section4/euroStat/nrg_cb_pem_page_spreadsheet_combustionFuels_nonRenewables.xlsx`
- renewable`../data/section4/euroStat/nrg_cb_pem_page_spreadsheet_combustionFuels_Renewables.xlsx`
- geothermal: `../data/section4/euroStat/nrg_cb_pem_page_spreadsheet_geothermal.xlsx`
- hydro: `../data/section4/euroStat/nrg_cb_pem_page_spreadsheet_hydro.xlsx`
- naturalGas: `../data/section4/euroStat/nrg_cb_pem_page_spreadsheet_naturalGas.xlsx`
- nuclear: `../data/section4/euroStat/nrg_cb_pem_page_spreadsheet_nuclear.xlsx`
- oil: `../data/section4/euroStat/nrg_cb_pem_page_spreadsheet_oil.xlsx`
- otherRenewables: `../data/section4/euroStat/nrg_cb_pem_page_spreadsheet_otherRenewables.xlsx`
- solar: `../data/section4/euroStat/nrg_cb_pem_page_spreadsheet_solar.xlsx`
- wind: `../data/section4/euroStat/nrg_cb_pem_page_spreadsheet_wind.xlsx`

## Step 2: Load Data into SQLite Database

We'll now read the downloaded CSV files, clean the data, and store them in an SQLite database.

### Python Code for Loading CSV Files into SQLite

```python
import pandas as pd
import sqlite3
import os

# Define file paths and dataset names
datasets = ['coal', 'nonRenewables', 'renewables', 'geothermal', 'hydro', 'naturalGas', 'nuclear', 'oil', 'otherRenewables', 'solar', 'wind']
data_path = os.path.expanduser('~/Downloads/')  # Path where the CSV files are downloaded
db_path = 'energy_data.db'  # SQLite database file

# Connect to SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

for dataset in datasets:
    # Load the CSV file into a pandas DataFrame
    csv_file = os.path.join(data_path, f'{dataset}.csv')
    df = pd.read_csv(csv_file, skiprows=1)  # Assuming the first row is metadata

    # Rename columns properly (example: replace unnamed columns with actual month names)
    df.columns = ['Country'] + [f'Month_{i+1}' for i in range(len(df.columns)-1)]

    # Store each DataFrame in a table named after the dataset (e.g., 'wind', 'geothermal')
    df.to_sql(dataset, conn, if_exists='replace', index=False)

    print(f"Data from {dataset}.csv loaded into the {dataset} table.")

# Commit and close the database connection
conn.commit()
conn.close()
```

### Explanation:
1. **Loading CSV Files**: We read each CSV file and load it into a pandas DataFrame, assuming the first row is metadata.
2. **Cleaning Data**: We rename the columns, as some columns might be unnamed.
3. **Storing in SQLite**: Each dataset is saved as a table in the SQLite database. For example, the `wind.csv` file will be saved in a table named `wind`.

---

## Step 3: Query and Analyze the Data

Now that we’ve stored the data in an SQLite database, we can query it using SQL and visualize it in Python. Let’s start by analyzing and plotting the wind energy production data.

### Python Code for Querying and Plotting Wind Data

```python
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd

# Reconnect to the SQLite database
conn = sqlite3.connect('energy_data.db')

# Query wind energy data
wind_query = """
SELECT * FROM wind;
"""

# Load the data into a pandas DataFrame
wind_data = pd.read_sql_query(wind_query, conn)

# Close the database connection
conn.close()

# Preview the data
print(wind_data.head())

# Example plotting wind energy production for Belgium
country = 'Belgium'

# Extract the data for Belgium (replace : and NaN values with 0 for simplicity)
wind_belgium = wind_data[wind_data['Country'] == country].fillna(0)

# Transpose to have months on x-axis
months = wind_belgium.columns[1:]  # Skip the 'Country' column
energy_values = wind_belgium.iloc[0, 1:]  # Skip the 'Country' column

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(months, energy_values, marker='o')
plt.title(f'Wind Energy Production in {country} (MWh)')
plt.xlabel('Months')
plt.ylabel('Energy Generated (MWh)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Explanation:
1. **Query Data**: We query the `wind` table and load the data into a pandas DataFrame.
2. **Data Cleaning**: We fill missing values (`NaN` or `:` in the dataset) with `0` for simplicity.
3. **Plotting**: We plot the wind energy production for Belgium over the months using `matplotlib`.

---

## Step 4: Plotting Energy Production for Multiple Countries

You can extend this example to plot data for multiple countries or compare the production of different renewable energy sources.

### Example Code to Plot Energy Production for Multiple Countries

```python
# Example for comparing wind energy production of Belgium, Germany, and Spain
countries = ['Belgium', 'Germany', 'Spain']
plt.figure(figsize=(10, 6))

for country in countries:
    country_data = wind_data[wind_data['Country'] == country].fillna(0)
    energy_values = country_data.iloc[0, 1:]  # Skip the 'Country' column
    
    plt.plot(months, energy_values, marker='o', label=country)

# Add labels, title, and legend
plt.title('Wind Energy Production in Belgium, Germany, and Spain (MWh)')
plt.xlabel('Months')
plt.ylabel('Energy Generated (MWh)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Explanation:
1. **Multiple Countries**: We loop through a list of countries (e.g., Belgium, Germany, Spain) and plot their wind energy production on the same graph for comparison.

---

## Workflow Summary

1. **Downloading and Loading Data**: We downloaded the CSV files from Eurostat and loaded them into an SQLite database.
2. **Querying the Database**: We queried specific tables from the database using SQL queries in Python.
3. **Visualizing the Data**: We plotted the renewable energy production data (wind in this case) using matplotlib to visualize the monthly energy production for different countries.

This approach allows you to easily handle large datasets, store them in a database, and perform complex analysis and visualization using Python.

---

## Summary

In this section, we provided a detailed step-by-step guide to integrating renewable energy production data stored in an SQLite database with Python for data analysis and visualization. The process includes downloading CSV files from the Eurostat website, loading them into an SQLite database using Python, and then querying the data for analysis and visualization.

## Key Points Covered:

1.	Data Download and Preparation: We downloaded the relevant CSV files for different renewable energy sources (wind, solar, geothermal, hydro, etc.) from Eurostat and stored them locally.
2.	Loading Data into SQLite: The downloaded CSV files were cleaned and stored in an SQLite database using the pandas and sqlite3 libraries in Python. Each dataset was stored in its own table within the database.
3.	Querying the Data: We queried the SQLite database for specific data (e.g., wind energy production) using SQL commands within Python.
4.	Visualizing Data: Using matplotlib, we created plots of the energy production data (e.g., wind energy production in Belgium) across different months.
5.	Comparison Across Countries: We extended the example to compare the energy production of multiple countries on a single plot, allowing for better insights into trends and differences in renewable energy generation.

## Lessons Learned

1.	**Handling Large Datasets**: By using `SQLite`, we can efficiently store and manage large datasets locally, allowing for easy querying and retrieval of data as needed.
2.	**Data Cleaning and Preparation**: Properly cleaning and formatting the data, such as renaming columns and filling missing values, is essential for accurate analysis.
3.	**SQL Integration with Python**: Using `SQL` within Python allows for powerful querying capabilities, which can be further extended by combining results with pandas for in-depth analysis.
4.	**Visualization**: Python’s `matplotlib` library makes it easy to create visual representations of the data, helping to uncover trends and insights from complex datasets.
5.	**Scalability**: The approach can be easily extended to more complex datasets, multiple energy sources, or multiple countries, providing a scalable solution for energy data analysis.
