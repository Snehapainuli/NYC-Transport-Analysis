NYC Transport Analysis Project

1. Overview
A Flask web app using Apache Spark to upload, explore, and analyze NYC transport data with SQL queries and machine learning.

2. Features
Upload CSV files with NYC transport data.
Automatic data cleaning and schema handling using Spark.
Interactive SQL query interface to explore the data.
Summary statistics and sample data preview.
Machine learning model training (regression/classification) using selected features and target.
Visualizations including Actual vs Predicted scatter plots.
Download prediction results as CSV.

3. Tech Stack
Python 3.x
Flask (Web framework)
Apache Spark (PySpark) for big data processing and ML
Pandas, Matplotlib, Seaborn for data handling and visualization

4. Setup Instructions
4.1 Prerequisites
Python 3.7 or higher
Java 8 or 11 (required for Spark)
Apache Spark 3.x installed and configured
pip (Python package installer)

4.2 Steps
Step 1: Clone the repository
git clone https://github.com/yourusername/nyc-transport-analysis.git
cd nyc-transport-analysis

Step 2: Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

Step 3: Install dependencies
pip install -r requirements.txt

Step 4: Set environment variables for Spark
Set SPARK_HOME to your Spark install directory.
Add $SPARK_HOME/bin to your system PATH.

Step 5: Run the Flask app
python app.py

5. Using the App
Open URL in your browser.
Upload a CSV transport dataset.
Explore data summary and run SQL queries.
Go to the ML form to select features and target column.
Train a model and view prediction results and plots.
Download predictions CSV.

6. Project Structure
nyc-transport-analysis/
│
├── app.py                 # Flask application code
├── spark_session.py       # Spark session setup (must be present)
├── model_utils.py         # ML model training and utility functions
├── templates/             # HTML templates (index.html, query.html, ml_form.html, result.html)
├── static/                # Static files (images, CSS, JS)
├── uploads/               # Uploaded CSV files (created at runtime)
├── results/               # Prediction output CSV files (created at runtime)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

8. Additional Notes
Ensure CSV datasets have headers matching expected column names 
Spark SQL is used for fast in-memory querying.
ML model training uses PySpark ML pipelines (model_utils.py).
Customize HTML templates in templates/ for UI changes.
