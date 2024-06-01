package com.szubd.rspalgos.classification

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
/**
 * @Author Lingxiang Zhao
 * @Date 2022/9/7 14:32
 * @desc
 */
object DecisionTreesSpark {

  var maxDepth: Int = 10
  var nodeSize: Int = 10

  def SparkDecisionTreesClassification(df: DataFrame, frac: Double=0.9):(PipelineModel, Double, Double) = {

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
    val decisionTree: DecisionTreeClassifier = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxDepth(maxDepth)
      .setMinInstancesPerNode(nodeSize)

    //建立工作流
    val pipeline = new Pipeline().setStages(Array(decisionTree))

    //开始计时
    val startTime = System.nanoTime
    //训练
    val model = pipeline.fit(train)

    //结束计时&计算训练耗时
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    if (frac < 0) {
      return (model, duration, 1)
    }
    //测试数据预测
    val result = model.transform(test)
    val rightCount= result.select("label", "prediction").rdd.filter(x => x.getInt(0) == x.getDouble(1).toInt).count()

    (model, duration, rightCount.toDouble / result.count())
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DecisonTrees-Classification-Spark").setMaster("yarn")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val df: DataFrame = spark.read.parquet("/user/caimeng/classification_50_2_0.54_5_64M.parquet").cache()
    val (model, duration, accuracy) = SparkDecisionTreesClassification(df)
    println("训练耗时："+ duration +"s")
    println("预测准确度" + accuracy)
    spark.stop()
  }
}
