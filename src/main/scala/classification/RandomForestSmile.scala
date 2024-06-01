package com.szubd.rspalgos.classification

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import smile.classification.{RandomForest, randomForest}
import smile.data.DataFrame
import smile.data.formula._
import smile.validation.metric.Accuracy

/**
 *
 */
object RandomForestSmile extends LogoClassifier[RandomForest] with Serializable{

  var nTrees: Int = 20
  var maxDepth: Int = 5

  def etl(rdd: RDD[Row]): RDD[(Array[Int], Array[Array[Double]])] = {
    rdd.glom().map(
      f => (
        f.map(r=>r.getInt(0)),
        f.map(r=>r.get(1).asInstanceOf[DenseVector].toArray)
      )
    )
  }

  def trainer(item: (Array[Int], Array[Array[Double]])): (RandomForest, Double) = {
    val trainLabel: Array[Array[Int]] = item._1.map(l => Array(l))
    val trainFeatures: Array[Array[Double]] = item._2
    val featureFrame: DataFrame = DataFrame.of(trainFeatures)  //特征列
    val labelFrame: DataFrame = DataFrame.of(trainLabel, "Y")  //标签列
    val formula: Formula = Formula.lhs("Y")     //创建Formula，设定除Y之外都是特征
    val trainFrame = featureFrame.merge(labelFrame)
    val startTime = System.nanoTime
    val forest: RandomForest = randomForest(formula, trainFrame, ntrees = this.nTrees, maxDepth=this.maxDepth)
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    (forest, duration)
  }

  def predictor(forest: RandomForest, features: Array[Array[Double]]): Array[Int] = {

    val testFeatures: Array[Array[Double]] = features
    val testFrame: DataFrame = DataFrame.of(testFeatures)

    return forest.predict(testFrame)
  }

  def accuracySmile(forest: RandomForest, item: (Array[Int], Array[Array[Double]])): Double = {

    estimator(predictor(forest, item._2), item._1)
  }

  def estimator(predict: Array[Int], label: Array[Int]): Double = {
//    predict.zip(label).count(item => item._1 == item._2).toDouble / predict.length
    Accuracy.of(label, predict)
  }

  def run(df : org.apache.spark.sql.DataFrame): Unit = {
    //    格式转换
    val rdd = etl(df.rdd)
    //    分训练和预测
    val Array(trainRDD: RDD[(Array[Int], Array[Array[Double]])], testRDD: RDD[(Array[Int], Array[Array[Double]])]) = rdd.randomSplit(Array(0.8, 0.2))
    //    训练
    val resultRDD = trainRDD.map(trainer)
    val modelRDD = resultRDD.map(_._1)
    //    预测
//    val predictRDD = modelRDD.cartesian(testRDD).map(f=>accuracySmile(f._1, f._2))
    val predictRDD = modelRDD.cartesian(testRDD).map(f=>estimator(predictor(f._1, f._2._2), f._2._1))
    //    获取模型在每个训练集分块上的accuracy
    val accuracies = predictRDD.collect()
    accuracies.foreach(println)
  }
}
