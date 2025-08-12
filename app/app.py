from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, to_timestamp
from pyspark.sql.types import DoubleType
from spark_session import spark
from model_utils import train_model

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    df = spark.read.csv(filepath, header=True, inferSchema=True)

    # Drop problematic columns with dots
    for col_to_drop in ["VehicleLocation.Latitude", "VehicleLocation.Longitude"]:
        if col_to_drop in df.columns:
            df = df.drop(col_to_drop)

    # Convert timestamp fields explicitly
    df = df.withColumn("ExpectedArrivalTime", to_timestamp("ExpectedArrivalTime"))
    df = df.withColumn("ScheduledArrivalTime", to_timestamp("ScheduledArrivalTime"))
    df = df.withColumn("RecordedAtTime", to_timestamp("RecordedAtTime"))

    # Cast known numeric columns
    cast_columns = ["OriginLat", "OriginLong", "DestinationLat", "DestinationLong", "DistanceFromStop"]
    for colname in cast_columns:
        if colname in df.columns:
            df = df.withColumn(colname, col(colname).cast(DoubleType()))

    df.createOrReplaceTempView("transport")
    return render_template("index.html", msg="✅ File uploaded and SQL table created.")

@app.route("/summary")
def summary():
    df = spark.table("transport")
    sample = df.limit(5).toPandas().to_html(classes="table table-bordered")
    stats = df.describe().toPandas().to_html(classes="table table-striped")
    return f"<h3>Sample Preview</h3>{sample}<br><h3>Summary</h3>{stats}"

@app.route("/query", methods=["GET", "POST"])
def query():
    result = ""
    if request.method == "POST":
        sql_query = request.form["sql_query"]
        try:
            df = spark.sql(sql_query)
            result = df.toPandas().to_html(classes="table table-striped")
        except Exception as e:
            result = f"<b>Error:</b> {str(e)}"
    return render_template("query.html", result=result)

@app.route("/ml_form")
def ml_form():
    df = spark.table("transport")

    # Only show these columns (and keep this order)
    allowed = [
        "DirectionRef",
        "PublishedLineName",
        "OriginName",
        "OriginLat",
        "OriginLong",
        "DestinationName",
        "DestinationLat",
        "DestinationLong",
        "DistanceFromStop",
        "ArrivalProximityText",
    ]

    # Keep only columns that actually exist in the current dataset
    columns = [c for c in allowed if c in df.columns]

    return render_template("ml_form.html", columns=columns)


@app.route("/predict", methods=["POST"])
def predict():
    target = request.form["target"]
    features = request.form.getlist("features")

    def quote(col_name):
        return f"`{col_name}`" if "." in col_name else col_name

    quoted_features = [quote(f) for f in features]
    quoted_target = quote(target)

    df = spark.table("transport").select(*quoted_features, quoted_target).dropna()

    model, pred_df, rmse, r2 = train_model(df, features, target)
    pdf = pred_df.toPandas()
    out_path = os.path.join(RESULT_FOLDER, "predictions.csv")
    pdf.to_csv(out_path, index=False)

    # Plot Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(pdf[target], pdf["prediction"], alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plot_path = os.path.join(STATIC_FOLDER, "prediction_plot.png")
    plt.savefig(plot_path)
    plt.close()

    return render_template("result.html", rmse=rmse, r2=r2, file="/download", plot_url="/static/prediction_plot.png")

@app.route("/download")
def download():
    return send_file("results/predictions.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
