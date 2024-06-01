package clustering

import com.szubd.rspalgos.clustering.ClusteringResult
import org.apache.spark.ml.clustering.{BisectingKMeans, BisectingKMeansModel, KMeans, KMeansModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf}

  /**
   * @Author Lingxiang Zhao
   * @Date 2022/9/19 11:06
   * @desc
   */
object SparkBisectingKMeans {

  var k: Int = 2
  var maxIterations: Int = 10

  def SparkBisectingKMeansClustering(
                             df: DataFrame,
                             testDf: DataFrame
                           ): ClusteringResult = {
    //开始计时
    val startTime = System.nanoTime

    //训练
    val bkm: BisectingKMeans = new BisectingKMeans().setK(k).setMaxIter(maxIterations)
    val model: BisectingKMeansModel = bkm.fit(df)

    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒

    if (testDf == null) {
      return new ClusteringResult(
        model, 1, duration, model.clusterCenters.map(_.toArray)
      )
    }

    val resultDf = model.transform(testDf)
    val labelCol = "label"
    val predictionCol = bkm.getPredictionCol
    var purity = resultDf
      .rdd
      .filter(r => r.getAs(labelCol) == r.getAs(predictionCol))
      .count().toDouble / resultDf.count()

    if (purity < 0.5) {
      purity = 1 - purity
    }

    new ClusteringResult(
      model, purity, duration, model.clusterCenters.map(_.toArray)
    )
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("BisectingKMeans-Clustering-Spark").setMaster("yarn")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val df: DataFrame = spark.read.parquet("/user/caimeng/classification_50_2_0.54_5_64M.parquet").cache()
    val Array(train, test) = df.randomSplit(Array(0.9, 0.1))
    val result = SparkBisectingKMeansClustering(train, test)

    println("训练耗时："+ result.getDuration() +"s")
    result.getCenters().foreach(v => printf("%s\n", v.toString))
    //    println("预测准确度" + accuracy)
    spark.stop()
  }
}
