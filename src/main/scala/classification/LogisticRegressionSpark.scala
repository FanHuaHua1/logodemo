package com.szubd.rspalgos.classification

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object LogisticRegressionSpark {

  var maxIter: Int = 500
  var regParam: Double = 0.1

  def SparkLinearRegressionClassification(df:DataFrame, frac: Double=0.9): (PipelineModel, Double, Double) = {
    //读取数
    var trainPart = frac
    var testPart = 1-frac
    if (frac < 0) {
      trainPart = -frac
      testPart = 1 - trainPart
    }
    printf("split: %f, %f\n", trainPart, testPart)
    val Array(train, test): Array[Dataset[Row]] = df.randomSplit(Array(trainPart, testPart))

    //建立模型
    val logisticRegression: LogisticRegression = new LogisticRegression()
      .setMaxIter(maxIter)
      .setRegParam(regParam)
      .setLabelCol("label").setFeaturesCol("features")
      .setFamily("binomial")

    //建立工作流
    val pipeline = new Pipeline().setStages(Array(logisticRegression))

    //开始计时
    val startTime = System.nanoTime

    //训练
    val logisticRegressionModel = pipeline.fit(train)

    //结束计时&计算训练耗时
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    if (frac < 0) {
      return (logisticRegressionModel, duration, 1)
    }
    //测试数据预测
    val result = logisticRegressionModel.transform(test)

    val rightCount= result.select("label", "prediction").rdd.filter(x => x.getInt(0) == x.getDouble(1).toInt).count()

    (logisticRegressionModel, duration, rightCount.toDouble / result.count())
    }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LogisticRegression-Classification-Spark").setMaster("yarn")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val df: DataFrame = spark.read.parquet("/user/caimeng/classification_50_2_0.54_5_64M.parquet").cache()
    val (model, duration, accuracy) = SparkLinearRegressionClassification(df)
    println("训练耗时："+ duration +"s")
    println("预测准确度" + accuracy)
    spark.stop()
  }
}
