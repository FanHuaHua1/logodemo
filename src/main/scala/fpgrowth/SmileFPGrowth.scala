package com.szubd.rspalgos.fpgrowth
import javafx.util.Pair

import java.util
import org.apache.spark.rdd.RDD
import smile.association.{ItemSet, fpgrowth}

import java.util.stream.{Collectors, Stream}
import scala.collection.mutable
import scala.collection.JavaConverters._
/**
 * @Author Lingxiang Zhao
 * @Date 2022/9/17 19:35
 * @desc
 */
object SmileFPGrowth {

  def etl(data: RDD[String]): RDD[Array[Array[Int]]] = {
    //把数据通过空格分割
    data.map(_.split(" ").map(_.toInt)).glom()
  }


  def runv(data: RDD[String], vote: Double, minsup: Int, path: String, modelPath: String): Unit = {
    val partitionCount: Int = data.getNumPartitions;
    //    格式转换
    val transaction: RDD[Array[Int]] = data.map((_: String).split(" ").map(_.toInt))
    transaction.count()
    val startTime = System.nanoTime
    //    获取每个分区的训练计时和stream结果
    val value: RDD[Stream[ItemSet]] = transaction.mapPartitions(arr => {
      val array: Array[Array[Int]] = arr.toArray
      val partitionRes: Stream[ItemSet] = fpgrowth(minsup, array)
      Iterator.single(partitionRes)
    })
    val value2: RDD[ItemSet] = value.mapPartitions((stream: Iterator[Stream[ItemSet]]) => {
      //迭代器里只有一个stream.Stream[ItemSet]
      val elem: Stream[ItemSet] = stream.next()
      val buf: mutable.Buffer[ItemSet] = elem.collect(Collectors.toList[ItemSet]).asScala
      buf.iterator
    })
    //将Itemset对象转成（频繁项集，出现次数）的KV对
    val value1: RDD[(String, Int)] = value2
      .filter(item => item.items.length > 1)
      .map((item: ItemSet) => (item.items.toList.sorted.mkString("{", ",", "}"), item.support))
      .cache()

    //
    val map: Map[String, Int] = value1.map(item => (item._1, 1)) //（频繁项集，1）
      .reduceByKey(_ + _) //（频繁项集，出现该频繁项集的分区数）
      .filter(f => f._2 >= partitionCount * vote) //（投票）
      .collect() //collect放到集合里
      .toMap

    val value3: RDD[(String, Int)] = value1
      .filter(f => map.contains(f._1))
      .combineByKey(v => v, (t: Int, v: Int) => t + v, (t: Int, v: Int) => t + v)
      .map(item => {
        (item._1, (item._2 / map(item._1)))
      })

    value3.map(x => x._1 + ": " + x._2).repartition(1).saveAsTextFile(path)
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    printf("Time spend: %f\n", duration)
    val value4: RDD[Pair[String, Int]] = value3.map(f => new Pair(f._1, f._2))
    value4.saveAsObjectFile(modelPath)
  }

  def runb(data: RDD[String], count: Long, elem: Double, ar: Int, path: String): Unit = {
    val partitionCount: Int = data.getNumPartitions;
    //    格式转换
    val transaction: RDD[Array[Int]] = data.map((_: String).split(" ").map(_.toInt)).cache()
    transaction.count()
    val startTime = System.nanoTime
    //    获取每个分区的训练计时和stream结果
    val value: RDD[Stream[ItemSet]] = transaction.mapPartitions(arr => {
      val array: Array[Array[Int]] = arr.toArray
      val partitionRes: Stream[ItemSet] = fpgrowth(ar, array)
      Iterator.single(partitionRes)
    })
    val value2: RDD[ItemSet] = value.mapPartitions((stream: Iterator[Stream[ItemSet]]) => {
      //迭代器里只有一个stream.Stream[ItemSet]
      val elem: Stream[ItemSet] = stream.next()
      val buf: mutable.Buffer[ItemSet] = elem.collect(Collectors.toList[ItemSet]).asScala
      buf.iterator
    })
    //用于广播
    val broadcastList2 = value2
      .filter((item: ItemSet) => item.items.length > 1)
      .map((item: ItemSet) => item.items.toList.sorted.mkString(","))
      .distinct()
      .collect()

    broadcastList2.foreach(println)
    val broadcastList: Array[List[Int]] = broadcastList2.map(_.split(",").map(_.toInt).toList)
    val value1: RDD[(String, Int)] = transaction.mapPartitions((arr: Iterator[Array[Int]]) => {
      val temp: Array[List[Int]] = util.Arrays.copyOf(broadcastList, broadcastList.length) //广播数组
      val set: Array[Set[Int]] = arr.map(_.toSet).toArray
      val partitionRes: Array[(String, Int)] = temp.map(items => { //List[Int]
        var count = 0
        for (orginalData <- set) { //List[Int]
          var b = true
          for (item <- items if b) { //Int
            if (!orginalData.contains(item)) {
              b = false
            }
          }
          if (b) {
            count = count + 1
          }
        }
        (items.mkString("{", ",", "}"), count)
      })
      partitionRes.iterator
    })

    value1.reduceByKey(_ + _).map(x => (x._1, x._2 * 1.0 / count)).filter(_._2 >= elem).map(x => x._1 + ": " + x._2).repartition(1).saveAsTextFile(path)
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    printf("Time spend: %f\n", duration)
  }
}
