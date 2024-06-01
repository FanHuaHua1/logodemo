package com.szubd.rspalgos

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import com.szubd.rspalgos.classification.{Entrypoint => ClsEntry}
import com.szubd.rspalgos.clustering.{Entrypoint => CltEntry}
import com.szubd.rspalgos.fpgrowth.{Entrypoint => FpgEntry}

object App {

  val KB = 1024
  val MB = 1024*KB
  val GB = 1024*MB
  val TB = 1024*GB.toLong

  var AppName: String = null
  var MaxExecutors: Int = 20

  lazy val conf: SparkConf = getConf()
  lazy val spark: SparkSession = getSpark()

  def getConf(): SparkConf = {
    var sparkConf = new SparkConf()
    sparkConf.setMaster("yarn")
    sparkConf.set("spark.dynamicAllocation.enabled", "true")
    sparkConf.set("spark.shuffle.service.enabled", "true")
    sparkConf.set("spark.dynamicAllocation.minExecutors", "1")
    sparkConf
  }

  def getSpark(): SparkSession = {

    val builder = SparkSession.builder().config(conf)

    if (AppName != null) {
      builder.appName(AppName)
    }

    builder.getOrCreate()
  }

  def main(args: Array[String]): Unit = {
    printf("Main args: %s\n", args.reduce((a, b) => a + " " + b))
    run(args)
  }

  def run(args: Array[String]): Unit = {
    if (args.length > 0) {
      args(0) match {
        case "clf" => ClsEntry.onArgs(spark, args.slice(1, args.length))
        case "clt" => CltEntry.onArgs(spark, args.slice(1, args.length))
        case "fpg" => FpgEntry.onArgs(spark, args.slice(1, args.length))
        case "--executors" => {
          //          MaxExecutors = args(1).toInt
          conf.set("spark.dynamicAllocation.maxExecutors", args(1))
          run(args.slice(2, args.length))
        }
        case "--conf" => {
          val confs = args(1).split(",")
          confs.foreach(c => {
            val kv = c.split("=")
            conf.set(kv(0), kv(1))
          })

          run(args.slice(2, args.length))
        }
        case _ => test(args)

      }

    }
  }

  def test(args: Array[String]): Unit = {
    printf("Unknown commands: %s\n", args.reduce(_ + ", " + _))
  }


}
