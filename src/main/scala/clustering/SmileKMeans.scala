package com.szubd.rspalgos.clustering

import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.RspContext._
import smile.clustering.KMeans
import smile.validation.metric.Accuracy

object SmileKMeans extends LogoClustering[KMeans] with Serializable{

  var k: Int = 2
  var maxIterations: Int = 100
  var tol: Double = 1.0E-4

  def etl(rdd: RDD[Row]): RDD[(LABEL, FEATURE)] = {
    rdd.glom().map(
      f => (
        f.map(_.getInt(0)),
        f.map(_.get(1).asInstanceOf[DenseVector].toArray)
      )
    )
  }

  def trainer(features: FEATURE): (KMeans, Double) = {
//    val trainFeatures: Array[Array[Double]] = item._2
    val startTime = System.nanoTime

    val kmeans: KMeans = KMeans.fit(features, k, maxIterations, tol)
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒

    (kmeans, duration)
  }

  def resample(model: KMeans): Array[Array[Double]] = {
    model.centroids
  }

//  def predictSmile(kmeans: KMeans, item: (Array[Int], Array[Array[Double]])): Double = {
//    val testLabel: Array[Int] = item._1
//    val testFeatures: Array[Array[Double]] = item._2
//    val prediction = Array[Int](testLabel.length)
//    for (i <- 0 until testLabel.length) {
//      prediction(i) = kmeans.predict(testFeatures(i));
//    }
//    val acc: Double = Accuracy.of(testLabel, prediction)
//    acc
//  }

  def purity(model: KMeans, test: (LABEL, FEATURE)): Double = {
    val predictions = test._2.map(model.predict)
    var p = predictions.zip(test._1)
      .count(r => r._1 == r._2)
      .toDouble / predictions.length
    if (p<0.5) {
      p = 1 - p
    }
    p
  }

  def showArray(vector: Array[Double]): Unit = {
    println((new DenseVector(vector)).toString)
  }

  def run(df : org.apache.spark.sql.DataFrame): Unit = {
    //    格式转换
    val rdd = etl(df.rdd)
    //    分训练和预测
//    val Array(trainRDD: RDD[(Array[Int], Array[Array[Double]])], testRDD: RDD[(Array[Int], Array[Array[Double]])]) = rdd.randomSplit(Array(0.8, 0.2))
    //    训练
    val modelsRDD = rdd.map(r => trainer(r._2))
    var resampledRDD = modelsRDD.map(item => resample(item._1)).flatMap(arr=>arr).coalesce(1).glom()
    var modelRDD = resampledRDD.map(trainer)
    var centers = modelRDD.map(r => resample(r._1)).collect()(0)
    centers.foreach(showArray)
  }


  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("KMeans-Clustering-Smile").setMaster("yarn")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val frame:  org.apache.spark.sql.DataFrame = spark.rspRead.parquet("/user/caimeng/classification_50_2_0.54_5_64M.parquet")
    run(frame)
    spark.stop()
  }
}
