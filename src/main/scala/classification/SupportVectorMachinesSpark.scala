package com.szubd.rspalgos.classification

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
object SupportVectorMachinesSpark {
  def SparkSupportVectorMachinesClassification(df:DataFrame, frac: Double=0.9): (LinearSVCModel, Double, Double) = {
    //读取数据
    val Array(train, test): Array[Dataset[Row]] = df.randomSplit(Array(frac, 1-frac))

    //建立模型
    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1).setTol(1e-6)

    //开始计时
    val startTime = System.nanoTime

    //训练
    val svmModel: LinearSVCModel = lsvc.fit(train)

    //结束计时&计算训练耗时
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒

    //测试数据预测
    val result = svmModel.transform(test)
    result.printSchema()
    //模型评估，预测准确性和错误率
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction")

    //获取预测准确度
    val accuracy: Double = evaluator.evaluate(result)

    (svmModel, duration, accuracy)
    }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SupportVectorMachines-Classification-Spark").setMaster("yarn")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val df: DataFrame = spark.read.parquet("/user/caimeng/classification_50_2_0.54_5_64M.parquet").cache()
    val (model, duration, accuracy) = SparkSupportVectorMachinesClassification(df)
    println("训练耗时："+ duration +"s")
    println("预测准确度" + accuracy)
    spark.stop()
  }
}
