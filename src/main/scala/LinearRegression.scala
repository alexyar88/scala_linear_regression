package my

import java.util.logging.Logger

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.abs
import breeze.stats.mean

import scala.collection.mutable.ArrayBuffer

class LinearRegression(lrLogger: Logger) {
  var lr: Double = 0.0001
  var w: DenseVector[Double] = DenseVector.fill(1)(0.2);
  var logger: Logger = lrLogger

  def fit(X: DenseMatrix[Double], y: DenseVector[Double], epochs: Int = 100, learningRate: Double = 0.0001): Unit = {
    val out = ArrayBuffer[String]()
    val ones = DenseMatrix.fill[Double](X.rows, 1)(1)
    val X_ = DenseMatrix.horzcat(ones, X)
    w = DenseVector.fill(X_.cols)(0.2)

    for (epoch <- 0 until epochs) {
      for (i <- 0 until X_.rows) {
        val grad = X_(i, ::) * (X_(i, ::) * w - y(i))
        w = w - learningRate * grad.t
      }
      val mae = mean(abs(y - X_ * w))
      logger.info(f"Epoch: $epoch, MAE=$mae%.2f")
    }
  }

  def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    val ones = DenseMatrix.fill[Double](X.rows, 1)(1)
    val X_ = DenseMatrix.horzcat(ones, X)
    X_ * w
  }


}
