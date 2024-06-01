package com.szubd.rspalgos.classification

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row}
import smile.classification.LogisticRegression
import smile.validation.metric.Accuracy

object LogisticRegressionSmile extends LogoClassifier[LogisticRegression] with Serializable {

  var maxIter: Int = 500
  var regParam: Double = 0.1

  def etl(rdd: RDD[Row]): RDD[(Array[Int], Array[Array[Double]])] = {

    rdd.glom().map(
      f => (
        f.map(r=>r.getInt(0)),
        f.map(r=>r.get(1).asInstanceOf[DenseVector].toArray)
      )
    )
  }

  /**
   *
   * @param sample  : 用于训练的数据类型
   * @return (model: M，trainTimeSeconds: Double)
   */
  override def trainer(sample: (LogisticRegressionSmile.LABEL, LogisticRegressionSmile.FEATURE)): (LogisticRegression, Double) = {
    val trainLabel: Array[Int] = sample._1
    val trainFeatures: Array[Array[Double]] = sample._2
    val startTime = System.nanoTime

    val model = LogisticRegression.fit(trainFeatures, trainLabel)
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    (model, duration)
  }

  /**
   *
   * @param prediction : 模型预测值
   * @param label      : 原始标签
   * @return accuracy: Double
   */
  override def estimator(prediction: LogisticRegressionSmile.LABEL, label: LogisticRegressionSmile.LABEL): Double = {
    Accuracy.of(label, prediction)
  }

  /**
   *
   * @param model    : M, 模型
   * @param features : FEATURE, 预测数据集
   * @return prediction: LABEL，预测结果
   */
  override def predictor(model: LogisticRegression, features: LogisticRegressionSmile.FEATURE): LogisticRegressionSmile.LABEL = {

    val testFeatures: Array[Array[Double]] = features

    return model.predict(testFeatures)
  }
}
