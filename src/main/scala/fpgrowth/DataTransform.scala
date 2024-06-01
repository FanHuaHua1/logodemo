package fpgrowth

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable
import org.apache.spark.sql.RspContext._
/**
 * @Author Lingxiang Zhao
 * @Date 2022/11/14 13:50
 * @desc
 */
object DataTransform {
  def main(args:Array[String]) {
    val conf = new SparkConf().setAppName("ParquetToText").setMaster("yarn")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    for (path <- args) {
      val data: DataFrame = spark.rspRead.parquet(path).cache()
      data.rdd.mapPartitions(r => r.map(_.get(0).asInstanceOf[mutable.WrappedArray[Int]].toArray.mkString(" "))).saveAsTextFile(path + ".txt")
    }

//    val value: RDD[String] = data.rdd.mapPartitions(r => r.map(_.get(0).asInstanceOf[mutable.WrappedArray[Int]].toArray.mkString(" ")))
//    val partitioner: Partitioner = new Partitioner {
//      override def numPartitions = 100
//
//      override def getPartition(key: Any) = key.asInstanceOf[Int]
//    }
//    value.map(row => (new Random().nextInt(100), row)).partitionBy(partitioner).saveAsTextFile("Items_10_5_RSP")

  }
}
