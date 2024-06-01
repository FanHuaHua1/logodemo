package com.szubd.rspalgos.clustering

import org.apache.spark.ml.util.MLWritable

class ClusteringResult(_model: MLWritable, _purity: Double, _duration: Double, _centers: Array[Array[Double]]) {

  def getModel(): MLWritable = _model

  def getPurity(): Double = _purity

  def getDuration(): Double = _duration

  def getCenters(): Array[Array[Double]] = _centers
}
