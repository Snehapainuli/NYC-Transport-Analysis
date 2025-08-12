from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, unix_timestamp

def train_model(df, features, target):
    stages = []
    updated_features = []

    print("\n🔍 Column Types:")
    for col_name in features:
        if dict(df.dtypes)[col_name] == 'string':
            print(f"{col_name}: {df.select(col_name).distinct().count()} distinct values")


        if dtype == 'string':
            indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed", handleInvalid="keep")
            stages.append(indexer)
            updated_features.append(col_name + "_indexed")
        elif dtype == 'timestamp':
            df = df.withColumn(col_name + "_ts", unix_timestamp(col(col_name)).cast("double"))
            updated_features.append(col_name + "_ts")
        else:
            updated_features.append(col_name)

    df = df.dropna(subset=features + [target])

    assembler = VectorAssembler(inputCols=updated_features, outputCol="features", handleInvalid="skip")
    print("\n🔍 Distinct category counts for string columns:")
    max_categories = 0
    for col_name in features:
        if dict(df.dtypes)[col_name] == 'string':
            distinct_count = df.select(col_name).distinct().count()
            print(f"{col_name}: {distinct_count}")
            max_categories = max(max_categories, distinct_count)
        
    print(f"\n✅ Largest category count: {max_categories}")


    # 🔁 Changed to GBTRegressor to avoid maxBins issues with DecisionTree-based models
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol=target,
        maxIter=50,
        maxDepth=5,
        maxBins=200
    )

    stages.extend([assembler, gbt])
    pipeline = Pipeline(stages=stages)

    model = pipeline.fit(df)
    predictions = model.transform(df)

    evaluator_rmse = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    return model, predictions.select(target, "prediction"), rmse, r2
