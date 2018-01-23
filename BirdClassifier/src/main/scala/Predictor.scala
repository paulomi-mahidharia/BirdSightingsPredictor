import ModelTrainer.{replaceXWithOne, toDouble}
import org.apache.spark.SparkConf
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{SparkSession, functions}

object Predictor {

  /**
    * value: label String
    * Returns: quivalent Double value with X replaced by 1.0
    */
  val replaceXWithOne = (value: String) => value match {
    case "?" => 1.0;
    case _ => value.toString.toDouble
  }

  /**
    * Input: Sequence of Strings
    * Output: Array of equivalent Double values with ? replaced by 0.0
    */
  val toDouble = (value: Seq[String]) => {
    value.map {
      case "?" => 0.0
      case v => v.toDouble
    }.toArray
  }

  val toInt = functions.udf((value: String) => value.toDouble.toInt)

  def main(args: Array[String]): Unit = {
    //Initialization of Spark Environment
    val sparkConf = new SparkConf()
    val spark = SparkSession.builder().appName("Species Predictor").config(sparkConf).getOrCreate()

    //Read the labeled csv file into a DataFrame
    val DF = spark.read.format("csv").option("header", "false").csv(args(0) + "/unlabeled.csv.bz2")

    //Define indexes of column which contain relavent features and the label
    val col_nums = Array(0, 2, 3, 26, 955, 956, 957, 958, 959, 960, 962, 963, 964, 965, 966, 967)

    //Select relevant features from the raw DF and create a new DF
    val observationDF = DF.selectExpr(col_nums.map(DF.columns(_)): _*).filter("_c0 not like '%SAMPLING%'")

    //Needed for toDF
    import spark.implicits._
    //Create a DF with LabeledPoint(label, Vetctor) for use with the classifier instance
    val labeledPointsWithSamplingEvent = observationDF.map(row => {
      (row.getString(0), replaceXWithOne(row.getString(3)),Vectors.dense(
        toDouble(Seq(row.getString(1), row.getString(2), row.getString(4),
          row.getString(5), row.getString(6), row.getString(7), row.getString(8),
          row.getString(9), row.getString(10), row.getString(11), row.getString(12),
          row.getString(13), row.getString(14), row.getString(15)))))
    }).toDF("sampling_event", "label", "features")

    //Load the trained model from file system
    val model = PipelineModel.load(args(1)+"/model")

    //Predict the labels for unlabeled data
    val predictions = model.transform(labeledPointsWithSamplingEvent)

    //Save the output in the desired format
    predictions
      .withColumn("prediction", toInt(predictions("prediction")))
      .selectExpr("sampling_event as SAMPLING_EVENT_ID","prediction as SAW_AGELAIUS_PHOENICEUS")
               .write.format("com.databricks.spark.csv")
               .option("header", "true")
               .save(args(1)+"/prediction")

    spark.stop()
  }

}
