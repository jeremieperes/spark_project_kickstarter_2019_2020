package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._


object Preprocessor {


  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    // Load data
    val path = System.getProperty("user.dir")

    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv(path + "/data/train_clean.csv")

    // Show dataframe and some insights
    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")
    df.show(10)
    df.printSchema

    // Cast numeric columns to Int
    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    // Get a statistical summary on Int columns
    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show

    // Look for data to clean
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)

    // Drop column disable_communication because it is almost always the same value
    val df2: DataFrame = dfCasted.drop("disable_communication")

    // Drop columns which cannot be known until the end of a campaign
    val dfNoFutur : DataFrame = df2.drop("backers_count","state_changed_at")

    // Create an UDF to clean the column country
    def cleanCountry(country: String, currency: String): String = {
      if (country == null)
        currency
      if (country.length!=2)
        null
      else
        country
    }
    val cleanCountryUdf = udf(cleanCountry _)

    // Create an UDF to clean the column currency
    def cleanCurrency(currency:String): String = {
      if (currency.length != 3)
        null
      else
        currency
    }
    val cleanCurrencyUdf = udf(cleanCurrency _)

    // Clean columns country and currency by creating new ones with UDFs previously created and dropping old ones
    val dfCountryCurrency = dfNoFutur
      .withColumn("country2",cleanCountryUdf($"country",$"currency"))
      .withColumn("currency2",cleanCurrencyUdf($"currency"))
      .drop("country","currency")

    // Clean all other columns
    val dfCleaned = dfCountryCurrency.filter($"final_status"===0 || $"final_status"===1)
      .withColumn("days_campaign",datediff(from_unixtime($"deadline"),from_unixtime($"launched_at")))
      .withColumn("launched_at",$"launched_at".cast("Float"))
      .withColumn("created_at",$"created_at".cast("Float"))
      .withColumn("hours_prepa",round(($"launched_at"-$"created_at")/3600,3))
      .drop("launched_at","deadline","created_at")
      .withColumn("name",lower($"name"))
      .withColumn("desc",lower($"desc"))
      .withColumn("keywords",lower($"keywords"))
      .withColumn("text",concat_ws(" ",$"name",$"desc",$"keywords"))
      .na.fill(Map("days_campaign" -> -1, "hours_prepa" -> -1,"goal" -> -1,"country2" -> "unknown","currency2" -> "unknown"))

    // Show final Dataframe
    dfCleaned.show(20)

    // Export the final Dataframe to parquet files
    dfCleaned.write.mode("overwrite").parquet(path + "/data/out")

  }
}
