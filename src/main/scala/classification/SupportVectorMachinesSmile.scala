package com.szubd.rspalgos.classification

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import smile.classification.{SVM, svm}
import smile.math.kernel.{LinearKernel}
import scala.math
import smile.validation.metric.Accuracy

object SupportVectorMachinesSmile extends LogoClassifier[SVM[Array[Double]]] with Serializable {

  var C: Double = 0.1
  var tol: Double = 1e-6
  var maxTrainSize: Long = 32 * 1024*1024
  var maxRecursive: Int = 3

  def etl(rdd: RDD[Row]): RDD[(Array[Int], Array[Array[Double]])] = {
    rdd.glom().map(
      f => (
        f.map(r=>{
          if(r.getInt(0) == 0)
            -1
          else
            1}),
        f.map(r=>r.get(1).asInstanceOf[DenseVector].toArray)
      )
    )
  }

  /**
   *
   * @param sample  : 用于训练的数据类型
   * @return (model: M，trainTimeSeconds: Double)
   */
  override def trainer(sample: (SupportVectorMachinesSmile.LABEL, SupportVectorMachinesSmile.FEATURE)): (SVM[Array[Double]], Double) = {
    val trainLabel: Array[Int] = sample._1
    val trainFeatures: Array[Array[Double]] = sample._2
    val startTime = System.nanoTime

    val model = batchTrain(trainLabel, trainFeatures, maxRecursive)
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    (model, duration)
  }


  /**
   *
   * @param prediction : 模型预测值
   * @param label      : 原始标签
   * @return accuracy: Double
   */
  override def estimator(prediction: SupportVectorMachinesSmile.LABEL, label: SupportVectorMachinesSmile.LABEL): Double = {
    Accuracy.of(label, prediction)
  }

  /**
   *
   * @param model    : M, 模型
   * @param features : FEATURE, 预测数据集
   * @return prediction: LABEL，预测结果
   */
  override def predictor(model: SVM[Array[Double]], features: SupportVectorMachinesSmile.FEATURE): SupportVectorMachinesSmile.LABEL = {

    val testFeatures: Array[Array[Double]] = features

    return model.predict(testFeatures)
  }

  def batchTrain(label: Array[Int], features: Array[Array[Double]], recursive: Int = 3): SVM[Array[Double]] = {
    printf("batch train, recursive=%d\n", recursive)
    val dimension = features(0).length
    val trainSize: Long = 32 * dimension * features.length
    var batch: Int = 1
    if(trainSize > maxTrainSize) {
      printf("trainSize(%d) > maxTrainSize(%d)\n", trainSize, maxTrainSize)
      batch = (trainSize / maxTrainSize).toInt + 1
    }
    printf("SVM batch = %d\n", batch)
    if(batch < 2){
      printf("Train from %d to %d\n", 0, label.length)
      return svm(features, label, new LinearKernel, C, tol)
    }
    val batchSize = label.length / batch

    if(recursive <= 0) {
      printf("No recursive chance left\n")
      return svm(features.slice(0, batchSize), label.slice(0, batchSize), new LinearKernel, C, tol)
    }

    val edges = Array.range(0, batch).map(i => (i*batchSize, (i+1)*batchSize))
    if (edges(edges.length-1)._2 < label.length) {
      edges(edges.length-1) = (edges(edges.length-1)._1, label.length)
    }

    val svs = edges.map(item => {
      val l = item._1
      val r = item._2
      printf("Train from %d to %d\n", l, r)
      val model = svm(features.slice(l, r), label.slice(l, r), new LinearKernel,  C, tol)
      var mean = model.weights.sum / model.weights.length
      var D = model.weights.map(i => i*i).sum - mean*mean
      var sd: Double = math.sqrt(D/model.weights.length)
      var minEdge = mean-sd
      var maxEdge = mean+sd
      printf(
        "Weights: mean=%f, var=%f, std=%f, range=[%f, %f], limits=[%f, %f]\n",
        mean, D, sd, minEdge, maxEdge, model.weights.min, model.weights.max
      )
      var sv = model.instances.zip(model.weights).filter(
//        w => (w._2 >= maxEdge) || (w._2 <= minEdge)
        _._2!=0
      ).map(item => {
        if(item._2>0){
          (item._1, 1)
        }else{
          (item._1, -1)
        }
      })
//      printf("Selected SV: %d\n", sv.length)
      sv

    }).flatMap(_.iterator)
    batchTrain(svs.map(_._2), svs.map(_._1), recursive-1)
//    svm(svs.map(_._1), svs.map(_._2), new LinearKernel, C, tol)
  }

  def modelStatus(svm: SVM[Array[Double]]): Array[Int] = {
    val instances = svm.instances()
    val weights = svm.weights()

    Array(instances.length, weights.length, weights.count(_!=0))
  }
}
