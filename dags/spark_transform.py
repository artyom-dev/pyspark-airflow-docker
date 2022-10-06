import os
import pandas as pd
import numpy as np
import airflow.utils.dates
from airflow import DAG
from airflow.operators.python import PythonOperator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def _spark_transform():
    dataset = "/opt/data/seattle.csv"
    spark = SparkSession.builder.master("local").appName("SparkByExamples.com").getOrCreate()

    # Configure environment for spark
    os.environ["DEBUSSY"] = "1"
    os.environ["PYSPARK_PYTHON"] = "python"

    # Create spark session and read data locally from csv
    df = spark.read.csv(dataset, header=True)

    # Since initially all columns in dataset was string, with aim of conducting
    # arithmetical operations I change some columns datatype to integer
    df = df.replace(float('NaN'), None)
    df = df.withColumn("Race/Ethnicity", col("Race/Ethnicity").cast('string')).withColumn("Sex", col("Sex").cast(
        'string')).withColumn("Department", col("Department").cast('string')).withColumn('Age', col('Age').cast(
        'integer')).withColumn('Hourly Rate', col('Hourly Rate').cast('integer')).withColumn('Regular/Temporary',
                                                                                             col('Regular/Temporary').cast(
                                                                                                 'string')).withColumn(
        'Employee Status', col('Employee Status').cast('string'))

    # In the case of NaN values I will fill in them with average of whole column
    mean_age_row = df.agg({'Age': 'avg'}).collect()
    mean_age = mean_age_row[0][0]
    df_final = df.na.fill(value=mean_age, subset=['Age'])

    # With goal to calculate correlations between features we need to encode categorical columns and obtain numericals instead.
    # For that I will use StringIndexer class which provided by PySpark
    df_strings = df_final.select(col('Race/Ethnicity'), col('Sex'), col('Department'), col('Regular/Temporary'),
                                 col('Employee Status'))
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").fit(df_strings) for column in
                df_strings.columns]
    pipeline = Pipeline().setStages(indexers)
    df_r = pipeline.fit(df_final).transform(df_final)

    # Now lets understand some initial statistics statistics
    df_final.groupBy(
        'Employee Status').count().show()  # A=Active; L=Leave of Absence; P=Paid Leave; S=Pending Separation/Layoff
    df_final.groupBy('Regular/Temporary').count().show()
    df_final.stat.corr('Age', 'Hourly Rate')  # T-Temporary occupied, R-Regular occupied

    print("For understanding distinct count of Races in Seattle city ")
    df_final.select(col('Race/Ethnicity')).distinct().show()

    df_final.groupBy('Race/Ethnicity').count().show()

    print("Count of jobs/departments ", df_final.select(col('department')).distinct().count())

    # Since at the moment of development I had problems with Hadoop integration with windows
    # and I didnt able to save Spark Dataframe as csv I decided to change dataframe format to pandas
    # and save csv through pandas. Since dataset isn't big enough saving in pandas doesnt create any issue.
    data = df_r.toPandas()
    data.to_csv("/opt/data/seattle_spark.csv")


def _correlation_and_visualisation_of_seattle():
    dataset = "/opt/data/seattle_spark.csv"
    df = pd.read_csv(dataset)
    features = ['Race/Ethnicity_index', 'Sex_index', 'Department_index', 'Regular/Temporary_index',
                'Employee Status_index']
    for feature in features:
        print(f'Correlation of {feature} with hourly rate is', stats.pointbiserialr(df[feature], df['Hourly Rate'])[0],
              " with P-value of ", stats.pointbiserialr(df[feature], df['Hourly Rate'])[1])

    fig = px.box(df, x="Sex", y="Hourly Rate", title='Box Plot of hourly rate')
    fig.show()

    group_labels = ['Distribution of Age']  # name of the dataset
    hist_data = [df['Age']]
    fig2 = ff.create_distplot(hist_data, group_labels)
    fig2.show()

    # fig3 = px.bar(df, x="Race/Ethnicity", y="Hourly Rate", color = "Race/Ethnicity", title="Salary distribution by races")
    # fig3.show()

    sorted_df = df.sort_values(by=['Hourly Rate'], ascending=False)
    sorted_df = sorted_df[sorted_df['Hourly Rate'] < 100]
    fig4 = px.histogram(sorted_df, x="Department", y='Hourly Rate',
                        title='Departments with hourly rate less 100 dollar')
    fig4.show()

    sorted_df2 = df.sort_values(by=['Hourly Rate'], ascending=False)
    sorted_df2['Hourly Rate'].between(100, 150, inclusive=True)
    # sorted_df2 = sorted_df2[sorted_df2['Hourly Rate'] >= 100 & sorted_df2['Hourly Rate'] < 150]
    fig5 = px.histogram(sorted_df2, x="Department", y='Hourly Rate',
                        title='Departments with hourly rate between 100 and 150 dollar')
    fig5.show()

    sorted_df3 = df.sort_values(by=['Hourly Rate'], ascending=False)
    sorted_df3 = sorted_df3[sorted_df3['Hourly Rate'] > 150]
    fig5 = px.histogram(sorted_df3, x="Department", y='Hourly Rate',
                        title='Departments with hourly rate more 150 dollar')
    fig5.show()

    ethnicity = df[['Race/Ethnicity', 'Hourly Rate', 'Department']]
    ethnicity_grouped = pd.DataFrame(ethnicity.groupby(['Race/Ethnicity', 'Department']).count()).reset_index()
    fig6 = px.scatter(ethnicity_grouped, x="Hourly Rate",
                      y='Department',
                      color="Race/Ethnicity",
                      size='Hourly Rate',
                      title="Scatterplot of hourly rate of each ethnicity based on department",
                      log_x=True, size_max=60)
    fig6.show()


def _regressionRF():
    """
    Regression using Random Forest Regressor with some hyperparameters
    """
    dataset = "/opt/data/seattle_spark.csv"
    data = pd.read_csv(dataset)
    data_new = data[['Race/Ethnicity_index', 'Sex_index', 'Department_index',
                     'Regular/Temporary_index', 'Employee Status_index', 'Age', 'Hourly Rate']]
    X = data_new[['Race/Ethnicity_index', 'Sex_index', 'Department_index',
                  'Regular/Temporary_index', 'Employee Status_index', 'Age']]
    y = data_new['Hourly Rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    regr = RandomForestRegressor(max_depth=7, random_state=0, n_estimators=150)
    regr.fit(X_train, y_train)
    prediction = regr.predict(X_test)
    y_true = np.array(y_test)
    prediction.shape, y_true.shape
    results = pd.DataFrame(data={"Y_true": y_true, "Y_pred": prediction})
    print(results)
    print(mean_squared_error(y_true, prediction))


with DAG(
        dag_id="spark_transform",
        start_date=airflow.utils.dates.days_ago(1),
        schedule_interval="@daily",
) as dag:
    run_spark_transform = PythonOperator(
        task_id="pyspark_transform",
        python_callable=_spark_transform,
        dag=dag
    )

    run_visualization_and_correlation = PythonOperator(
        task_id="run_visualization",
        python_callable=_correlation_and_visualisation_of_seattle,
        dag=dag
    )

    run_ml = PythonOperator(
        task_id="run_ML",
        python_callable=_regressionRF,
        dag=dag
    )

run_spark_transform >> run_visualization_and_correlation >> run_ml
