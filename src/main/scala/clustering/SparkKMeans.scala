package com.szubd.rspalgos.clustering

import org.apache.spark.{SparkConf, mllib}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
//import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.ml.linalg.{DenseVector => MLDenseVector}
import org.apache.spark.rdd.RDD
import smile.validation.metric.Accuracy
/**
 * @Author Lingxiang Zhao
 * @Date 2022/9/10 16:48
 * @desc
 */
object SparkKMeans {

  var k: Int = 2
  var maxIterations: Int = 100
  var tol: Double = 1.0E-4

  def SparkKMeansClustering(
                             df: DataFrame,
                             testDf: DataFrame
                           ): ClusteringResult = {
    //训练数据
    //val trainLabel: Array[Int] = train.select("label").collect().map(row => row.mkString.toInt)

//    val trainFeatures: RDD[mllib.linalg.Vector] = df
//      .select("features")
//      .rdd
//      .map(r => DenseVector.fromML(r.get(0).asInstanceOf[MLDenseVector]))
//      .map(row => Vectors.dense(row.mkString.replaceAll("\\[|\\]", "").split(",").map(_.toDouble)))

    //开始计时
    val startTime = System.nanoTime

    val kmeans = new KMeans().setK(k).setMaxIter(maxIterations).setTol(tol)
    val model = kmeans.fit(df)
    //训练
//    val model: KMeansModel = KMeans.train(trainFeatures, k, maxIterations)

    //结束计时&计算训练耗时
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒

    if (testDf == null) {
      return new ClusteringResult(
        model, 1, duration, model.clusterCenters.map(_.toArray)
      )
    }
    val resultDf = model.transform(testDf)
    val labelCol = "label"
    val predictionCol = kmeans.getPredictionCol
    var purity = resultDf
      .rdd
      .filter(r => r.getAs(labelCol) == r.getAs(predictionCol))
      .count().toDouble / resultDf.count()

    if (purity < 0.5) {
      purity = 1 - purity
    }

//    (model, duration, model.clusterCenters.map(_.toArray))
    new ClusteringResult(
      model, purity, duration, model.clusterCenters.map(_.toArray)
    )
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Kmeans-Clustering-Spark").setMaster("yarn")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val df: DataFrame = spark.read.parquet("/user/caimeng/classification_50_2_0.54_5_64M.parquet").cache()
    val Array(train, test) = df.randomSplit(Array(0.9, 0.1))
    val result = SparkKMeansClustering(train, test)

    println("训练耗时："+ result.getDuration() +"s")
    result.getCenters().foreach(v => printf("%s\n", v.toString))
//    println("预测准确度" + accuracy)
    spark.stop()
  }
}
