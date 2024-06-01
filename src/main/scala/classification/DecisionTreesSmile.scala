package com.szubd.rspalgos.classification

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import smile.classification.{DecisionTree, cart}
import smile.data.DataFrame
import smile.data.formula._
import smile.validation.metric.Accuracy

object DecisionTreesSmile extends LogoClassifier[DecisionTree] with Serializable {

  var maxDepth: Int = 10
  var nodeSize: Int = 10

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
  override def trainer(sample: (DecisionTreesSmile.LABEL, DecisionTreesSmile.FEATURE)): (DecisionTree, Double) = {
    val trainLabel: Array[Array[Int]] = sample._1.map(l => Array(l))
    val trainFeatures: Array[Array[Double]] = sample._2
    val featureFrame: DataFrame = DataFrame.of(trainFeatures)  //特征列
    val labelFrame: DataFrame = DataFrame.of(trainLabel, "Y")  //标签列
    val formula: Formula = Formula.lhs("Y")     //创建Formula，设定除Y之外都是特征
    val startTime = System.nanoTime
    val trainFrame = featureFrame.merge(labelFrame)
    val tree: DecisionTree = cart(formula, trainFrame, maxDepth = this.maxDepth, nodeSize = this.nodeSize)
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    (tree, duration)
  }

  /**
   *
   * @param prediction : 模型预测值
   * @param label      : 原始标签
   * @return accuracy: Double
   */
  override def estimator(prediction: DecisionTreesSmile.LABEL, label: DecisionTreesSmile.LABEL): Double = {
    Accuracy.of(label, prediction)
  }

  /**
   *
   * @param model    : M, 模型
   * @param features : FEATURE, 预测数据集
   * @return prediction: LABEL，预测结果
   */
  override def predictor(model: DecisionTree, features: DecisionTreesSmile.FEATURE): DecisionTreesSmile.LABEL = {
    val testFeatures: Array[Array[Double]] = features
    val testFrame: DataFrame = DataFrame.of(testFeatures)
    return model.predict(testFrame)
  }
}
