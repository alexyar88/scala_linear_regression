package my

import java.io.File
import java.util.logging.{FileHandler, Logger, SimpleFormatter}

import breeze.stats.{mean, variance}
import breeze.numerics.abs
import breeze.linalg.DenseVector
import breeze.linalg.{DenseMatrix, DenseVector, csvread}

object Utils {
  def mae(y: DenseVector[Double], y_hat: DenseVector[Double]): Double = {
    mean(abs(y_hat - y))
  }

  def getXyFromCsvPath(path: String, targetCol: Int): Tuple2[DenseMatrix[Double], DenseVector[Double]] = {
    val csvFile: File = new File(path)
    val data: DenseMatrix[Double] = csvread(csvFile, skipLines = 1)
    val maxColNum = data.cols - 1
    val cols = for (i <- 0 to maxColNum if i != targetCol) yield i
    val X = data(::, cols).toDenseMatrix
    val y: DenseVector[Double] = data(::, targetCol)
    (X, y)
  }

  def getLogger(name: String, outputPath: String): Logger = {
    System.setProperty(
      "java.util.logging.SimpleFormatter.format",
      "%1$tF %1$tT %4$s %5$s%6$s%n"
    )

    val logger = Logger.getLogger(name)
    val handler = new FileHandler(outputPath)
    val formatter = new SimpleFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger
  }
}
