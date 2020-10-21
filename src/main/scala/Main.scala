import my.LinearRegression
import my.Utils.{getXyFromCsvPath, mae, getLogger}

object Main {

  def main(args: Array[String]): Unit = {

    val logger = getLogger(name = "linear_regression", outputPath = "linear_regression.log")

    var dataTrain = ""
    var dataTest = ""
    var targetCol = 0
    args.sliding(2, 2).toList.collect {
      case Array("--data-train", argDataTrain: String) => dataTrain = argDataTrain
      case Array("--data-test", argDataTest: String) => dataTest = argDataTest
      case Array("--target-column", argTargetCol: String) => targetCol = argTargetCol.toInt
    }


    logger.info(f"Loading train data: $dataTrain")

    val Xy_train = getXyFromCsvPath(path = dataTrain, targetCol = targetCol)
    val X_train = Xy_train._1
    val y_train = Xy_train._2

    logger.info(f"Start training")
    val lr = new LinearRegression(logger)
    lr.fit(X_train, y_train)
    val y_hat_train = lr.predict(X_train)
    val mae_train = mae(y_hat_train, y_train)
    logger.info(f"MAE train: $mae_train%.2f")

    logger.info(f"Loading validation data: $dataTest")
    val Xy_test = getXyFromCsvPath(dataTest, targetCol)
    val X_test = Xy_test._1
    val y_test = Xy_test._2
    val y_hat_test = lr.predict(X_test)
    val mae_test = mae(y_hat_test, y_test)
    logger.info(f"MAE test: $mae_test%.2f")

  }
}
