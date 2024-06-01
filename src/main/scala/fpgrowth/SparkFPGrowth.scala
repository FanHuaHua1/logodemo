package com.szubd.rspalgos.fpgrowth

import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
/**
 * @Author Lingxiang Zhao
 * @Date 2022/9/17 19:35
 * @desc
 */
object SparkFPGrowth {
  def etl(data: RDD[String]): RDD[Array[String]] = {
    //把数据通过空格分割
    data.map(_.split(" "))
  }

  def run(data: RDD[String], partitions:Int, path:String): Unit = {
    //    格式转换
   val data1: RDD[Array[String]] = etl(data.repartition(partitions)).cache()
    //val data1: RDD[Array[String]] = etl(data).cache()
    data1.count();
    val startTime = System.nanoTime
    val fpg = new FPGrowth().setMinSupport(0.15)
    val model = fpg.run(data1)
    val value: RDD[FPGrowth.FreqItemset[String]] = model.freqItemsets.filter(f => f.items.length > 1)
    value.repartition(1).saveAsTextFile(path)
    val duration = (System.nanoTime - startTime) * 0.000000001  //System.nanoTime为纳秒，转化为秒
    printf("Time spend: %f\n", duration)
  }
}
