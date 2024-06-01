package com.szubd.rspalgos.clustering

import clustering.bmutils.LOGOBisectingKMeans
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}


/**
 * @Author Lingxiang Zhao
 * @Date 2022/9/19 10:33
 * @desc
 */
object SmileBisectingKMeans extends LogoClustering[LOGOBisectingKMeans] with Serializable{

  var k: Int = 2
  var maxIterations: Int = 2

  def etl(rdd: RDD[Row]): RDD[(LABEL, FEATURE)] = {
    rdd.glom().map(
      f => (
        f.map(_.getInt(0)),
        f.map(_.get(1).asInstanceOf[DenseVector].toArray)
      )
    )
  }

  def trainer(features: FEATURE): (LOGOBisectingKMeans, Double) = {
    val bisecting = new LOGOBisectingKMeans(k, maxIterations);

    val startTime = System.nanoTime

   bisecting.clustering(features)

    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒

    (bisecting, duration)
  }

  def resample(model: LOGOBisectingKMeans): Array[Array[Double]] = {
    model.centroids
  }

  def purity(model: LOGOBisectingKMeans, test: (LABEL, FEATURE)): Double = {
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
    //    训练
    val modelsRDD = rdd.map(r => trainer(r._2))
    var resampledRDD = modelsRDD.map(item => resample(item._1)).flatMap(arr=>arr).coalesce(1).glom()
    var modelRDD = resampledRDD.map(trainer)
    var centers = modelRDD.map(r => resample(r._1)).collect()(0)
    centers.foreach(showArray)
  }


  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("BisectingKMeans-Clustering-Smile").setMaster("yarn")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val frame: org.apache.spark.sql.DataFrame = spark.read.parquet("/user/caimeng/classification_50_2_0.54_5_64M.parquet")
    run(frame)
    spark.stop()
  }

}
