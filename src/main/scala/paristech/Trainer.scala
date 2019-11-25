package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, HashingTF, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, Tokenizer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel,RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()
    import spark.implicits._


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    // Load data
    val path = System.getProperty("user.dir")

    val df: DataFrame = spark
      .read
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .parquet(path + "/data/prepared_trainingset")

    df.show(5)

    df.printSchema

    // Stage 1 : Get words
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // Stage 2 : Remove stop words
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("words")

    // Stage 3 : TF
    val vectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("rawFeatures")

    // Stage 4 : IDF
    val idf = new IDF()
      .setInputCol(vectorizer.getOutputCol)
      .setOutputCol("tfidf")

    // Stage 5 : convert country2
    val countryIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    // Stage 6 : convert currency2
    val currencyIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    // Stage 7-8 : One-Hot encoding for country and currency
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array(countryIndexer.getOutputCol, currencyIndexer.getOutputCol))
      .setOutputCols(Array("country_onehot", "currency_onehot"))


    // Stage 9 : Vector features assembler
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    // Stage 10 : Classification model - Logistic Regression
    println("Training a Logistic Regression model")

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    // Pipeline creation
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer,remover,vectorizer,idf,countryIndexer,currencyIndexer,encoder,assembler,lr))

    // Data splitting
    val Array(trainingData, testData) = df.randomSplit(Array(0.9, 0.1),42)

    // Training model
    val model = pipeline.fit(trainingData)

    // Make predictions
    val dfWithSimplePredictions = model.transform(testData)

    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    // Evaluate the model
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    // Check F1 score
    val f1 = evaluator.evaluate(dfWithSimplePredictions)
    println("F1 score with Logistic Regression:" + f1)

    //  Classification model - Random Forest
    println("Training a Random Forest model")

    val rf = new RandomForestClassifier()
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")

    val rfPipeline = new Pipeline()
      .setStages(Array(tokenizer,remover,vectorizer,idf,countryIndexer,currencyIndexer,encoder,assembler,rf))
    // Training model
    val rfModel = rfPipeline.fit(trainingData)

    // Make predictions
    val dfWithPredictionsRf = rfModel.transform(testData)

    dfWithPredictionsRf.groupBy("final_status", "predictions").count.show()

    // Evaluate the model
    val rfEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    // Check F1 score
    val rfF1 = rfEvaluator.evaluate(dfWithPredictionsRf)
    println("F1 score with Random Forest :" + rfF1)

    // Trying to find the best hyperparameters on Logistic Regression
    println("Cross validation on Logistic Regression")
    // Building a param grid on different values of regParam
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8,10e-6,10e-4,10e-2))
      .addGrid(vectorizer.minDF, Array(55.0,75.0,95.0))
      .build()

    // Cross validation
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    // Training model
    val cvModel = cv.fit(trainingData)

    // Make predictions
    val dfWithPredictionsCv = cvModel.transform(testData)

    dfWithPredictionsCv.groupBy("final_status", "predictions").count.show()

    // Evaluate the model
    val cvEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val cvF1 = cvEvaluator.evaluate(dfWithPredictionsCv)
    println("F1 score with Logistic Regression + Cross Validation :" + cvF1)

    // Saving the best model (ie Logistic Regression with Cross validation)
    cvModel.write.overwrite().save(path + "/model")


  }
}
