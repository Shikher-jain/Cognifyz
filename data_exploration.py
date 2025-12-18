import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
file_path = 'Dataset.csv'
df = pd.read_csv(file_path)

# 1. Explore dataset shape
num_rows, num_cols = df.shape
print(f"Rows: {num_rows}, Columns: {num_cols}")

# 2. Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Handle missing values (example: fill with mode for categorical, mean for numerical)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# 3. Data type conversion (example: ensure 'Aggregate rating' is float)
df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'], errors='coerce')

# 4. Analyze target variable distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['Aggregate rating'], bins=20, kde=True)
plt.title('Aggregate Rating Distribution')
plt.xlabel('Aggregate rating')
plt.ylabel('Count')
plt.show()

# Check for class imbalance
print(df['Aggregate rating'].value_counts())

# 5. Descriptive statistics for numerical columns
print("Descriptive statistics:")
print(df.describe())

# 6. Distribution of categorical variables
for cat_col in ['Country Code', 'City', 'Cuisines']:
    print(f"\nValue counts for {cat_col}:")
    print(df[cat_col].value_counts().head(10))

# 7. Top cuisines and cities with highest number of restaurants
print("\nTop cuisines:")
print(df['Cuisines'].value_counts().head(10))

print("\nTop cities:")
print(df['City'].value_counts().head(10))

# --- Geospatial Analysis ---
# 1. Visualize restaurant locations on a map
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    fig = px.scatter_mapbox(
        df,
        lat='Latitude',
        lon='Longitude',
        color='Aggregate rating',
        hover_name='City',
        mapbox_style='open-street-map',
        title='Restaurant Locations by Aggregate Rating',
        height=500
    )
    fig.show()
else:
    print('Latitude and Longitude columns not found in the dataset.')

# =============================
# LEVEL 2: Advanced Analysis & Feature Engineering
# =============================

# Task 1: Table Booking and Online Delivery
for col in ['Has Table booking', 'Has Online delivery']:
    if col in df.columns:
        percent = df[col].str.lower().eq('yes').mean() * 100
        print(f"Percentage of restaurants with {col}: {percent:.2f}%")
    else:
        print(f"Column '{col}' not found in the dataset.")

if 'Has Table booking' in df.columns:
    avg_rating_table = df.groupby(df['Has Table booking'].str.lower())['Aggregate rating'].mean()
    print("\nAverage rating by Table Booking:")
    print(avg_rating_table)
else:
    print("Column 'Has Table booking' not found in the dataset.")

if 'Has Online delivery' in df.columns and 'Price range' in df.columns:
    delivery_by_price = df.groupby(['Price range', df['Has Online delivery'].str.lower()]).size().unstack(fill_value=0)
    print("\nOnline delivery availability by Price Range:")
    print(delivery_by_price)
else:
    print("Required columns for online delivery by price range not found.")

# Task 2: Price Range Analysis
if 'Price range' in df.columns:
    most_common_price = df['Price range'].mode()[0]
    print(f"Most common price range: {most_common_price}")
    avg_rating_by_price = df.groupby('Price range')['Aggregate rating'].mean()
    print("\nAverage rating for each price range:")
    print(avg_rating_by_price)
    if 'Rating color' in df.columns:
        color_by_price = df.groupby('Rating color')['Aggregate rating'].mean()
        top_color = color_by_price.idxmax()
        print(f"\nRating color with highest average rating: {top_color}")
    else:
        print("'Rating color' column not found.")
else:
    print("'Price range' column not found.")

# Task 3: Feature Engineering
if 'Restaurant Name' in df.columns:
    df['Name Length'] = df['Restaurant Name'].astype(str).apply(len)
    print('Added feature: Name Length')
else:
    print('Column "Restaurant Name" not found.')

if 'Address' in df.columns:
    df['Address Length'] = df['Address'].astype(str).apply(len)
    print('Added feature: Address Length')
else:
    print('Column "Address" not found.')

for col in ['Has Table booking', 'Has Online delivery']:
    if col in df.columns:
        df[f'{col} (bin)'] = df[col].str.lower().map({'yes': 1, 'no': 0})
        print(f'Added binary feature: {col} (bin)')
    else:
        print(f'Column "{col}" not found.')

print(df.head())

# =============================
# LEVEL 3: Predictive Modeling, Customer Preferences & Visualization
# =============================

# Task 1: Predictive Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

features = df.select_dtypes(include=[float, int]).drop(columns=['Aggregate rating'], errors='ignore')
target = df['Aggregate rating']
mask = target.notnull()
X = features[mask]
y = target[mask]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE={mse:.3f}, R2={r2:.3f}")

# Task 2: Customer Preference Analysis
if 'Cuisines' in df.columns:
    cuisine_ratings = df.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
    print('Average rating by cuisine (top 10):')
    print(cuisine_ratings.head(10))
else:
    print('Column "Cuisines" not found.')

if 'Cuisines' in df.columns and 'Votes' in df.columns:
    cuisine_votes = df.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False)
    print('\nMost popular cuisines by total votes (top 10):')
    print(cuisine_votes.head(10))
else:
    print('Required columns for cuisine votes not found.')

if 'Cuisines' in df.columns:
    high_rating_cuisines = cuisine_ratings[cuisine_ratings >= 4.0]
    print('\nCuisines with average rating >= 4.0:')
    print(high_rating_cuisines)
else:
    print('Column "Cuisines" not found.')

# Task 3: Data Visualization
# 1. Distribution of ratings (histogram)
plt.figure(figsize=(8, 4))
sns.histplot(df['Aggregate rating'], bins=20, kde=True)
plt.title('Distribution of Aggregate Ratings')
plt.xlabel('Aggregate rating')
plt.ylabel('Count')
plt.show()

# 2. Average ratings by cuisine (bar plot, top 10)
if 'Cuisines' in df.columns:
    top_cuisines = df['Cuisines'].value_counts().head(10).index
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Cuisines', y='Aggregate rating', data=df[df['Cuisines'].isin(top_cuisines)], ci=None, estimator=np.mean)
    plt.title('Average Ratings of Top 10 Cuisines')
    plt.xlabel('Cuisine')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.show()

# 3. Average ratings by city (bar plot, top 10)
if 'City' in df.columns:
    top_cities = df['City'].value_counts().head(10).index
    plt.figure(figsize=(10, 5))
    sns.barplot(x='City', y='Aggregate rating', data=df[df['City'].isin(top_cities)], ci=None, estimator=np.mean)
    plt.title('Average Ratings of Top 10 Cities')
    plt.xlabel('City')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.show()

# 4. Relationship between price range and rating
if 'Price range' in df.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x='Price range', y='Aggregate rating', data=df)
    plt.title('Aggregate Rating by Price Range')
    plt.xlabel('Price Range')
    plt.ylabel('Aggregate Rating')
    plt.show()

# 2. Distribution of restaurants across countries and top cities
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Country Code', order=df['Country Code'].value_counts().index)
plt.title('Number of Restaurants by Country')
plt.xlabel('Country Code')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
top_cities = df['City'].value_counts().head(15).index
sns.countplot(data=df[df['City'].isin(top_cities)], x='City', order=top_cities)
plt.title('Number of Restaurants in Top 15 Cities')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.show()

# 3. Correlation between location and rating
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    print('Correlation between Latitude and Aggregate rating:', df[['Latitude', 'Aggregate rating']].corr().iloc[0,1])
    print('Correlation between Longitude and Aggregate rating:', df[['Longitude', 'Aggregate rating']].corr().iloc[0,1])
else:
    print('Latitude and Longitude columns not found in the dataset.')
