import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * This Program trains the classification model using Random Forest Classifier
  * on labeled data
  */
object ModelTrainer {

  /**
    * value: label String
    * Returns: equivalent Double value with X replaced by 1.0
    * If value if > 1.0, return 1.0
    */
  val replaceXWithOne = (value: String) => value match {
    case "X" => 1.0;
    case _ => if (value.toDouble > 1.0) 1.0 else value.toDouble
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

  def main(args: Array[String]): Unit = {
    //Initialization of Spark Environment
    val sparkConf = new SparkConf()
    val spark = SparkSession.builder().appName("Bird Model Trainer").config(sparkConf).getOrCreate()

    //Read the labeled csv file into a DataFrame
    val DF = spark.read.format("csv").option("header", "false").csv(args(0) + "/labeled.csv.bz2")

    //Define indexes of column which contain relavent features and the label
    val col_nums = Array(0, 2, 3, 26, 955, 956, 957, 958, 959, 960, 962, 963, 964, 965, 966, 967)

    //Select relevant features from the raw DF and create a new DF
    val observationDF = DF.selectExpr(col_nums.map(DF.columns(_)): _*).filter("_c0 not like '%SAMPLING%'")

    //Needed for toDF
    import spark.implicits._
    //Create a DF with LabeledPoint(label, Vetctor) for use with the classifier instance
    val labeledPointsWithSamplingEvent =  observationDF.mapPartitions(iterator => {
      iterator.map(row => {
        (row.getString(0), replaceXWithOne(row.getString(3)), Vectors.dense(
          toDouble(Seq(row.getString(1), row.getString(2), row.getString(4),
            row.getString(5), row.getString(6), row.getString(7), row.getString(8),
            row.getString(9), row.getString(10), row.getString(11), row.getString(12),
            row.getString(13), row.getString(14), row.getString(15)))))
      })
    }).toDF("sampling_event", "label", "features")

    //Split the data into training and validation data
    val Array(trainingData, testData) = labeledPointsWithSamplingEvent.randomSplit(Array(0.8, 0.2))

    //Index the labels
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel")
      .fit(labeledPointsWithSamplingEvent)

    //Initialize the Random Forest Classifier with required parameters
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setNumTrees(70)
      .setMinInstancesPerNode(12)
      .setImpurity("gini")

    //Define a pipeline to train the model with following steps: LabelIndexer ->RF  Classifier
    val pipeline = new Pipeline().setStages(Array(labelIndexer, rf))

    //Generate a model by fitting on the training data
    val model = pipeline.fit(trainingData)

    //Save the model for use in the predictor job that predicts labels for unlabeled data
    model.save(args(1) + "/model")

    //Evaluate the accuracy on validation data
    val predictions = model.transform(testData)

    // Select (prediction, true label) and compute test error.
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")

    //Get the accuracy and log to console
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy) + "AND Accuracy = " + accuracy)

    spark.stop()
  }

}
