# Restaurant Data Analysis Project

This project explores, analyzes, and models restaurant data using Python. The workflow is organized into three levels, each with specific tasks:

## Level 1: Data Exploration & Descriptive Analysis
- Explore the dataset and identify the number of rows and columns.
- Check for missing values and handle them accordingly.
- Perform data type conversion if necessary.
- Analyze the distribution of the target variable ("Aggregate rating") and identify any class imbalances.
- Calculate basic statistical measures (mean, median, standard deviation, etc.) for numerical columns.
- Explore the distribution of categorical variables like "Country Code", "City", and "Cuisines".
- Identify the top cuisines and cities with the highest number of restaurants.

## Level 2: Advanced Analysis & Feature Engineering
- Table Booking and Online Delivery: Analyze the percentage, ratings, and price range relationships.
- Price Range Analysis: Most common price range, average ratings, and color with highest average rating.
- Feature Engineering: Extract new features (e.g., name/address length, binary encodings).

## Level 3: Predictive Modeling, Customer Preferences & Visualization
- Predictive Modeling: Build and evaluate regression models (linear, tree, forest) for aggregate rating.
- Customer Preference Analysis: Analyze cuisine/rating relationships, most popular cuisines, and high-rated cuisines.
- Data Visualization: Visualize distributions, compare ratings by cuisine/city, and explore feature relationships.

## How to Run

## Using a Virtual Environment (Recommended)
It is recommended to use a Python virtual environment to manage dependencies and avoid conflicts:

1. Create a virtual environment:
	```bash
	python -m venv venv
	```
2. Activate the virtual environment:
	- On Windows:
	  ```bash
	  venv\Scripts\activate
	  ```
	- On macOS/Linux:
	  ```bash
	  source venv/bin/activate
	  ```
3. Install requirements:
	```bash
	pip install -r requirements.txt
	```
4. Open and run the notebook `check.ipynb` in Jupyter or VS Code.
5. Follow the notebook cells for stepwise analysis and modeling.

## Requirements
See `requirements.txt` for all dependencies.

## Files
- `check.ipynb`: Main analysis notebook
- `Dataset.csv`: Input data
- `requirements.txt`: Python dependencies
- `README.md`: Project overview (this file)

---

**Author:** Shikher Jain