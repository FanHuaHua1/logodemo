package com.szubd.rspalgos.classification

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import javafx.util.Pair

import org.apache.spark.ml.util.MLWritable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.RspContext._
import org.apache.spark.rsp.RspRDD
import spire.ClassTag

import scala.collection.mutable.WrappedArray
import scala.collection.mutable.{Map => MutablMap}
import scala.util.Random

object Entrypoint {

  /**
   *
   * @param spark
   * @param args: Array[String], spark|logo commands...
   */
  def onArgs(spark: SparkSession, args: Array[String]): Unit = {
    args(0) match {
      case "spark" => onFitSpark(spark, args)
      case "logo" => onFitLogo(spark, args)
      case "logof" => onFitLogo(spark, args, false)
      case "testlogo" => onTestLogo(spark, args)
      case "sparkShuffle" => onFitSparkShuffle(spark, args)
      case "sparkSample" => onFitSparkShuffleSample(spark, args)
      case "sparkSampleOnly" => onFitSparkShuffleSampleOnly(spark, args)
      case "logoShuffle" => fitLogoShuffle(spark, args, true)
      case _ => printf("Unknown type: %s\n", args(0))
    }
  }

  def onTestLogo(spark: SparkSession, args: Array[String]): Unit = {
    val algo = args(1)
    val sourceFile = args(2)
    var frac: Double = 0.9
    if(args.length > 3) {
      frac = args(3).toDouble
      if (frac > 1) {
        frac = 0.9
      }
    }

    val df = spark.rspRead.parquet(sourceFile)
    algo match {
      case "SVM" => LogoApp.run(df, SupportVectorMachinesSmile, frac)
      case "DT" => LogoApp.run(df, DecisionTreesSmile, frac)
      case "RF" => LogoApp.run(df, RandomForestSmile, frac)
      case "LR" => LogoApp.run(df, LogisticRegressionSmile, frac)
    }


  }

  /**
   *
   * @param spark
   * @param args: Array[String], spark algorithm dataFile partitionFile fraction sizes...
   *  algorithm: DT|LR|RF
   *  dataFile: 实验数据文件
   *  partitionFile: 分块记录文件，主要记录对应文件不同大小的实验需要取那些分块
   *  fraction: 训练集数据比例：0~1
   *  sizes: 需要运算的数据大小，单位GB，支持50:50:500，可接受多参数输入例如: 50 100 150
   *
   *  sample: clf spark DT rspclf50_8000.parquet SubPartitionIds8000.parquet 0.9 50 100 150 200 250 300 350 400 450 500
   */
  def onFitSpark(spark: SparkSession, args: Array[String]): Unit = {
    val algo = args(1)
    val sourceFile = args(2)
    val partitionFile = args(3)
    val fraction = args(4).toDouble
    val sizes = args.slice(5, args.length).map(_.toInt)

    fitSpark(spark, algo, sourceFile, partitionFile, fraction, sizes)

  }

  /**
   *
   * @param spark
   * @param args : Array[String], logo algorithm dataFile partitionFile subs tests predicts tails sizes...
   *  algorithm: DT|LR|RF
   *  dataFile: 实验数据文件
   *  partitionFile: 分块记录文件，主要记录对应文件不同大小的实验需要取那些分块
   *  subs: rsp计算所用的数据比列， 0 ~ 1
   *  tests: 模型训练测试集比例, 0 ~ 1
   *  predicts: 模型预测数据集块数
   *  tails: 模型集成过程中的头尾筛选比例
   *  sizes: 需要运算的数据大小，单位GB，支持50:50:500，可接受多参数输入例如: 50 100 150
   *
   *  sample: clf logo DT rspclf50_8000.parquet SubPartitionIds8000.parquet 0.05 0.1 1 0.05 50 100 150 200 250 300 350 400 450 500
   * @param useScore
   */
  def onFitLogo(spark: SparkSession, args: Array[String], useScore: Boolean=true): Unit = {
    val algo = args(1)
    val sourceFile = args(2)
    val partitionFile = args(3)
    val subs = args(4).toDouble
    val tests = args(5).toDouble
    val predicts = args(6).toDouble
    val tails = args(7).toDouble
    val sizes = args.slice(8, args.length).map(_.toInt)
    fitLogo(spark, algo, sourceFile, partitionFile, subs, tests, predicts, tails, sizes, useScore)
  }

  def fitSpark(spark: SparkSession,
               algo: String,
               sourceFile: String,
               partitionFile: String,
               fraction: Double,
               sizes: Array[Int]): Unit = {

    printf("fitAlgo: algorithm = %s\n", algo)
    printf("fitAlgo: sourceFile = %s\n", sourceFile)
    printf("fitAlgo: partitionFile = %s\n", partitionFile)
    printf("fitAlgo: fraction = %f\n", fraction)
    var rdf = spark.rspRead.parquet(sourceFile)
    var pdf = spark.read.parquet(partitionFile)
    var parts = pdf.collect().map(
      r => (r.getInt(0), r.get(1).asInstanceOf[WrappedArray[Int]].toArray)
    )
    var jobs: Array[(Int, Array[Int])] = null
    if (sizes.size > 0) {
      printf("fit algo: sizes = %s\n", sizes.map(_.toString).reduce(_ + ", " + _))
      var sizeMapper = parts.toMap

      //      jobs = sizes.filter(s => sizeMapper.contains(s)).map(s => (s, sizeMapper(s)))
      jobs = sizes.map(s => (s, sizeMapper.getOrElse(s, Array.range(0, s))))
    } else {
      jobs = parts
    }

    for ((size, partitions) <- jobs) {
      var trainName = "train(size=%d)".format(size)
      var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
      var modelName = "%s_%s_%d_%s.ml".format(algo, sourceFile, size, beginTime)
      //      var partitions = sizeMapper(size)
      //      var rdd = getSubRdd(rdf.rdd, partitions)
      var rdd = rdf.rdd.getSubPartitions(partitions)

      var df = spark.createDataFrame(rdd, rdf.schema)

      algo match {
        case "DT" => runSpark(trainName, modelName, df, fraction,
          //          DecisionTrees.SparkDecisionTreesClasscification)
          DecisionTreesSpark.SparkDecisionTreesClassification)
        case "LR" => runSpark(trainName, modelName, df, fraction,
          //          LinearRegression.SparkLinearRegressionClassification)
          LogisticRegressionSpark.SparkLinearRegressionClassification)
        case "RF" => runSpark(trainName, modelName, df, fraction,
          //          RandomForest.SparkRandomForestClasscification)
          RandomForestSpark.SparkRandomForestClassification)
        case "SVM" => runSpark(trainName, modelName, df, fraction,
          SupportVectorMachinesSpark.SparkSupportVectorMachinesClassification)

      }
    }

  }

  def onFitSparkShuffle(spark: SparkSession, args: Array[String]): Unit = {
    sparkShuffle(
      spark, args(1), args(2), args(3).toDouble, args(4).toDouble, args(5).toDouble,
      args.slice(6, args.length).map(_.toDouble)
    )
  }

  def onFitSparkShuffleSample(spark: SparkSession, args: Array[String]): Unit = {
    sparkShuffle(
      spark, args(1), args(2), args(3).toDouble, args(4).toDouble,args(5).toDouble,
      args.slice(6, args.length).map(_.toDouble * 0.05)
    )
  }

  def onFitSparkShuffleSampleOnly(spark: SparkSession, args: Array[String]): Unit = {
    sparkShuffleSample(
      spark, args(1), args(2), args(3).toDouble, args(4).toDouble, args(5).toDouble,
      args.slice(6, args.length).map(_.toDouble)
    )
  }


  /**
   * clf sparkShuffle algo sourceFile fraction partitionsUnit size1 size2 ...
   *
   * @param spark
   * @param algo: String, 算法 DT|LR|RF|SVM
   * @param sourceFile: String, 数据文件
   * @param fraction: Double, 训练集比例
   * @param partitionsUnit: Double, 单位数据块数, 计算取块数 = (size * partitionsUnit).toInt()
   * @param clusterPercentage : Double, 该模拟集群单位比例
   * @param sizes: Array[Int], 训练数据集大小
   */
  def sparkShuffle(spark: SparkSession,
                   algo: String,
                   sourceFile: String,
                   fraction: Double,
                   partitionsUnit: Double,
                   clusterPercentage: Double,
                   sizes: Array[Double]): Unit = {

    printf("fitAlgo: algorithm = %s\n", algo)
    printf("fitAlgo: sourceFile = %s\n", sourceFile)
    printf("fitAlgo: fraction = %f\n", fraction)
    var rdf = spark.rspRead.parquet(sourceFile)

    var jobs: Array[(Double, Array[Int])] = null
    val partitionsList = List.range(0, rdf.rdd.getNumPartitions)
    if (sizes.length > 0){
      jobs = sizes.map(s => (s, Random.shuffle(partitionsList).toArray))
    }else{
      jobs = Array((0, Array()))
    }

    for ((size, partitions) <- jobs) {
      //      printf("size=%d, partitions=[%s]\n", size, partitions.map(_.toString).reduce(_ + ", " + _))
      var trainName = "train(size=%f)".format(size)
      var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
      var modelName = "%s_%s_%f_%s.ml".format(algo, sourceFile, size, beginTime)
      //      var partitions = sizeMapper(size)
      var rdd = rdf.rdd
      if (size > 0) {
        var partitionCount = (Math.floor(size * partitionsUnit * clusterPercentage)).toInt
        printf("partitionCount=%d\n", partitionCount)
        printf("partitions=%s\n", partitions.slice(0, partitionCount).toList)
        rdd = rdd.getSubPartitions(partitions.slice(0, partitionCount))
      }

      var df = spark.createDataFrame(rdd, rdf.schema)

      algo match {
        case "DT" => runSpark(trainName, modelName, df, fraction,
          //          DecisionTrees.SparkDecisionTreesClasscification)
          DecisionTreesSpark.SparkDecisionTreesClassification)
        case "LR" => runSpark(trainName, modelName, df, fraction,
          //          LinearRegression.SparkLinearRegressionClassification)
          LogisticRegressionSpark.SparkLinearRegressionClassification)
        case "RF" => runSpark(trainName, modelName, df, fraction,
          //          RandomForest.SparkRandomForestClasscification)
          RandomForestSpark.SparkRandomForestClassification)
        case "SVM" => runSpark(trainName, modelName, df, fraction,
          SupportVectorMachinesSpark.SparkSupportVectorMachinesClassification)

      }
    }

  }


  /**
   * clf sparkShuffle algo sourceFile fraction partitionsUnit size1 size2 ...
   *  仅采样不建模
   * @param spark
   * @param algo           : String, 算法 DT|LR|RF|SVM （无所谓用不上）
   * @param sourceFile     : String, 数据文件
   * @param fraction       : Double, 训练集比例 （无所谓用不上）
   * @param partitionsUnit : Double, 单位数据块数, 计算取块数 = (size * partitionsUnit).toInt()
   * @param clusterPercentage : Double, 该模拟集群单位比例
   * @param sizes          : Array[Int], 训练数据集大小
   */
  def sparkShuffleSample(spark: SparkSession,
                   algo: String,
                   sourceFile: String,
                   fraction: Double,
                   partitionsUnit: Double,
                   clusterPercentage: Double,
                   sizes: Array[Double]): Unit = {

    printf("fitAlgo: sourceFile = %s\n", sourceFile)
    var rdf = spark.rspRead.parquet(sourceFile)

    var jobs: Array[(Double, Array[Int])] = null
    val partitionsList = List.range(0, rdf.rdd.getNumPartitions)
    if (sizes.length > 0) {
      jobs = sizes.map(s => (s, Random.shuffle(partitionsList).toArray))
    } else {
      jobs = Array((0, Array()))
    }

    for ((size, partitions) <- jobs) {
      //      printf("size=%d, partitions=[%s]\n", size, partitions.map(_.toString).reduce(_ + ", " + _))
      var trainName = "train(size=%f)".format(size)
      var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
      var modelName = "%s_%s_%f_%s.ml".format(algo, sourceFile, size, beginTime)
      //      var partitions = sizeMapper(size)
      var rdd = rdf.rdd
      if (size > 0) {
        var partitionCount = (Math.floor(size * partitionsUnit * clusterPercentage)).toInt
        printf("partitions=%s\n", partitions.slice(0, partitionCount).toList)
        rdd = rdd.getSubPartitions(partitions.slice(0, partitionCount))
      }
      rdd.first()
      var sampleTrainName = "train(size=%f)/sample\n".format(size)
      printf(sampleTrainName)
      val sampleStartTime = System.nanoTime
      val sampleRdd: RDD[Row] = rdd.sample(false, 0.05)
      sampleRdd.count()
      val sampleDuration = (System.nanoTime - sampleStartTime) * 0.000000001
      printf("Time spend: %f\n", sampleDuration)
    }
  }


  def runSpark(trainName: String, modelName: String,
               df: DataFrame, fraction: Double,
               function: (DataFrame, Double) => (MLWritable, Double, Double)): Unit = {
    printf("%s start\n", trainName)
    var (model, duration, accuracy) = function(df, fraction)
    printf("%s finished\n", trainName)
    printf("Time spend: %f\n", duration)
    printf("%s Accuracy: %f\n", trainName, accuracy)
    model.write.save(modelName)
  }

  def fitLogo(spark: SparkSession,
              algo: String,
              sourceFile: String,
              partitionFile: String,
              subs: Double, tests: Double, predicts: Double,
              tails: Double,
              sizes: Array[Int], useScore: Boolean=true): Unit = {
    var rdf = spark.rspRead.parquet(sourceFile)
    var pdf = spark.read.parquet(partitionFile)

    var parts = pdf.collect().map(
      r => (r.getInt(0), r.get(1).asInstanceOf[WrappedArray[Int]].toArray)
    )
    var jobs: Array[(Int, Array[Int])] = null
    if (sizes.length > 0) {
      printf("fit algo: sizes = %s\n", sizes.map(_.toString).reduce(_ + ", " + _))
      var sizeMapper = parts.toMap
      //      jobs = sizes.filter(s => sizeMapper.contains(s)).map(s => (s, sizeMapper(s)))
      jobs = sizes.map(s => (s, sizeMapper.getOrElse(s, Array.range(0, s))))
    } else {
      jobs = parts
    }

    var predictBlocks = math.ceil(predicts).toInt
    var predictSample = predicts / predictBlocks
    for ((size, partitions) <- jobs) {
      //      printf("size=%d, partitions=[%s]\n", size, partitions.map(_.toString).reduce(_ + ", " + _))
      var trainName = "train(size=%d)".format(size)
      var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
      //      var modelName = "%s_%s_%d_%s.ml".format(algo, sourceFile, size, beginTime)
      //      var partitions = sizeMapper(size)
      //      var rdd = getSubRdd(rdf.rdd, partitions)

      var trainParts = (partitions.length * subs).toInt
      //      printf("trainSize = %d * %f = %d\n", partitions.length, subs, trainParts)
      //      printf("trainSize=%d, parts=[%s]\n", trainParts, partitions.slice(0, trainParts).map(_.toString).reduce(_ + ", " + _))
      //      var testParts = trainParts + 1
      var predictParts = trainParts + predictBlocks
      var trainRdd = rdf.rdd.getSubPartitions(partitions.slice(0, trainParts))
      //      var testRdd = rdf.rdd.getSubPartitions(partitions.slice(trainParts, testParts)).sample(false, tests)
      var predictRdd: RDD[Row] = rdf.rdd.getSubPartitions(partitions.slice(trainParts, predictParts))
      if (predictSample < 1) {
        predictRdd = predictRdd.sample(false, predictSample)
      }
      printf("%s\n", trainName)
      algo match {
        case "RF" => runLogo(trainName, trainRdd, predictRdd, RandomForestSmile, tails, useScore, "")
        case "DT" => runLogo(trainName, trainRdd, predictRdd, DecisionTreesSmile, tails, useScore,"")
        case "LR" => runLogo(trainName, trainRdd, predictRdd, LogisticRegressionSmile, tails, useScore,"")
        case "SVM" => {
          trainRdd = new RspRDD[Row](trainRdd.repartition(trainRdd.getNumPartitions*8).cache())
          runLogo(trainName, trainRdd, predictRdd, SupportVectorMachinesSmile, tails, useScore,"")
        }
      }

    }
  }

//  def fitLogoShuffle(spark: SparkSession, args: Array[String], useScore: Boolean): Unit = {
//    logoShuffle(
//      spark, args(1), args(2),
//      args(3).toDouble, args(4).toDouble, args(5).toDouble,
//      args(6).toDouble, args.slice(7, args.length).map(_.toInt),
//      useScore
//    )
//  }

  /**
   * 带模型地址
   * @param spark
   * @param args
   * @param useScore
   */
  def fitLogoShuffle(spark: SparkSession, args: Array[String], useScore: Boolean): Unit = {
    logoShuffle(
      spark, args(1), args(2), args(3),
      args(4).toDouble, args(5).toDouble, args(6).toDouble,
      args(7).toDouble, args.slice(8, args.length).map(_.toInt),
      useScore
    )
  }
  /**
   *
   * @param spark
   * @param algo: String, 算法 DT|LR|RF|SVM
   * @param sourceFile: String, 数据文件
   * @param subs: Double, rsp取块比例
   * @param predicts: Double, 预测集块数
   * @param tails: Double, 头尾筛选比例
   * @param partitionsUnit: Double, 单位数据块数, 计算取块数 = (size * partitionsUnit).toInt()
   * @param sizes: Array[Int], 训练数据集大小
   * @param useScore
   */
  def logoShuffle(spark: SparkSession,
                  algo: String,
                  sourceFile: String,
                  modelPath:String,
                  subs: Double, predicts: Double,
                  tails: Double,
                  partitionsUnit: Double,
                  sizes: Array[Int], useScore: Boolean=true): Unit = {
    var inputPath = modelPath
    var rdf = spark.rspRead.parquet(sourceFile)

    var jobs: Array[(Int, Array[Int])] = null
    val partitionsList = List.range(0, rdf.rdd.getNumPartitions)
    if (sizes.length > 0){
      jobs = sizes.map(s => (s, Random.shuffle(partitionsList).toArray))
    }else{
      jobs = Array((0, Array()))
    }

    var predictBlocks = math.ceil(predicts).toInt
    var predictSample = predicts / predictBlocks
    var predictRdd: RDD[Row] = null
    for ((size, partitions) <- jobs) {
      //      printf("size=%d, partitions=[%s]\n", size, partitions.map(_.toString).reduce(_ + ", " + _))
      var trainName = "train(size=%d)".format(size)
      var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
      //      var modelName = "%s_%s_%d_%s.ml".format(algo, sourceFile, size, beginTime)
      var partitionCount = (size * partitionsUnit)
      var trainParts = (partitionCount * subs).toInt

      var predictParts = trainParts + predictBlocks
      printf(
        "size=%d, trainParts(%d)=%s, predictParts(%d)=%s\n",
        size,
        trainParts, partitions.slice(0, trainParts).toList,
        predictParts-trainParts, partitions.slice(trainParts, predictParts).toList
      )
      var trainRdd = rdf.rdd.getSubPartitions(partitions.slice(0, trainParts))
      //      var testRdd = rdf.rdd.getSubPartitions(partitions.slice(trainParts, testParts)).sample(false, tests)
      if(predictParts > trainParts) {
        predictRdd = rdf.rdd.getSubPartitions(partitions.slice(trainParts, predictParts))

        if (predictSample < 1) {
          predictRdd = predictRdd.sample(false, predictSample)
        }
      }else{
        predictRdd = null
      }
      printf("%s\n", trainName)
      algo match {
        case "RF" => runLogo(trainName, trainRdd, predictRdd, RandomForestSmile, tails, useScore, inputPath)
        case "DT" => runLogo(trainName, trainRdd, predictRdd, DecisionTreesSmile, tails, useScore, inputPath)
        case "LR" => runLogo(trainName, trainRdd, predictRdd, LogisticRegressionSmile, tails, useScore, inputPath)
        case "SVM" => {
          trainRdd = new RspRDD[Row](trainRdd.repartition(trainRdd.getNumPartitions*8).cache())
          runLogo(trainName, trainRdd, predictRdd, SupportVectorMachinesSmile, tails, useScore, inputPath)
        }
      }
      inputPath = inputPath + "_" + size
    }
  }


  def aggScore(accs: Array[(Double, Int)]): Double = {

    return accs.map(item => item._1 * item._2).sum / accs.map(_._2).sum
  }

  def labelScore(predictions: Array[Int], score: Double): Array[Double] = {
    return predictions.map(_*score)
  }

  def aggPredictionBinary(
                           param: (Long, (Iterable[(Long, Double, Array[Int])], Int))
                         ): (Long, Array[Int]) = {
    val result = new Array[Double](param._2._2)
    var total: Double = 0
    for ((_, factor: Double, prediction: Array[Int]) <- param._2._1) {
      total += factor
      for(i <- 0 until param._2._2) {
        result(i) = result(i) + prediction(i) * factor
      }
    }
    (param._1, result.map(_/total).map(_.round.toInt))
  }

  def votePrediction(param: (Long, (Iterable[(Long, Array[Int])], Int))): (Long, Array[Int]) = {
    val labels = param._2._1.map(_._2).toArray
    val result = new Array[Int](param._2._2)
    val members = labels.length
    val counts = MutablMap[Int, Int]()
    for (i <- 0 until param._2._2) {
      for (m <- 0 until members) {
        counts(labels(m)(i)) = counts.getOrElse(labels(m)(i), 0) + 1
      }
      result(i) = counts.maxBy(_._2)._1
      counts.clear()
    }

    (param._1, result)
  }


  def runLogo[
    M: ClassTag // Model
  ](trainName: String, trainRdd: RDD[Row], predictRdd: RDD[Row],
    classifier: LogoClassifier[M],
    tail: Double, useScore: Boolean=true, modelPath: String): Unit = {
    val beginTime = System.nanoTime

    //    var modelRdd = classifier.etl(trainRdd).map(
    //      classifier.trainer
    //    ).zipWithIndex().map(item => (item._2, item._1._1, item._1._2))

    var modelRdd = classifier.etl(trainRdd).map(
      sample => {
        val trainSize = (sample._1.length * 0.9).toInt
        val trainSample = (sample._1.slice(0, trainSize), sample._2.slice(0, trainSize))
        val testSample = ((sample._1.slice(trainSize, sample._1.length), sample._2.slice(trainSize, sample._1.length)))
        val (model, duration) = classifier.trainer(trainSample)
        val accuracy = classifier.estimator(classifier.predictor(model, testSample._2), testSample._1)
        (model, duration, accuracy)
      }
    )

    //    modelRdd.count()
    var valuedModels: RDD[(M, Double)] = null

    if ( useScore ) {
      val integrateStart = System.nanoTime
      val factors = modelRdd.map(_._3).collect().sorted
      val mcount = factors.length
      printf("Model count: %d\n", mcount)
      var tcount = (mcount * tail).toInt
      if(tcount < 1) {
        tcount = 1
      }
      printf("Tail count: %d\n", tcount)
      val (minAcc, maxAcc) = (factors(tcount), factors(mcount-tcount-1))
      printf("Score range: (%f, %f)\n", minAcc, maxAcc)
      //      printf("Range count: %d\n", )
      //      printf("Time integrate: %f\n", (System.nanoTime - integrateStart) * (1e-9))
      valuedModels = modelRdd.filter(item => minAcc <= item._3 && item._3 <= maxAcc).map(item => (item._1, item._3))
    } else {
      valuedModels = modelRdd.map(item => (item._1, 1.0))
    }

    val endTime = System.nanoTime()
    printf("Time spend: %f\n", (endTime - beginTime) * (1e-9) )
    //
    val value: RDD[Pair[M, Double]] = valuedModels.map(f => new Pair(f._1, f._2))
    value.saveAsObjectFile(modelPath)
    if (predictRdd == null) {
      printf("%s Accuracy: %f\n", trainName, 1.0)
      return
    }
    printf("ValuedModel count: %d\n", valuedModels.count())

    val predictWithIndex = classifier.etl(predictRdd).zipWithIndex()
    val predicts = predictWithIndex.map(item => (item._2, item._1, item._1._1.length))
    val prediction = valuedModels.cartesian(predicts).map(
      item => (item._2._1, classifier.predictor(item._1._1, item._2._2._2))
    ).groupBy(_._1)
    val sizeRDD = predicts.map(item => (item._1, item._3))
    val rspPredict = prediction.join(sizeRDD).map(votePrediction)
    val indexedLabels = predictWithIndex.map(item => (item._2, item._1._1))
    val rspAcc = rspPredict.join(indexedLabels).map(
      item => (classifier.estimator(item._2._1, item._2._2), item._2._1.length)
    )
    val acc = rspAcc.map(item => item._1 * item._2).sum / rspAcc.map(_._2).sum
    //    val result = rspAcc.collect()
    //    val acc = result.map(item => item._1 * item._2).sum / result.map(_._2).sum
    printf("%s Accuracy: %f\n", trainName, acc)
  }

}
