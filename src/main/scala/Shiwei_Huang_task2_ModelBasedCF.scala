import java.io.{File, PrintWriter}

import scala.math._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

object ModelBasedCF {


  def main (args: Array[String]): Unit = {

    var conf = new SparkConf()
    conf.setAppName("ModelBasedCF")
    conf.setMaster("local[4]")

    //record start-time
    val start_time = System.currentTimeMillis()

    val sc = new SparkContext (conf)
    val input1 = sc.textFile(args(0))
    //val input1 = sc.textFile("./data/video_small_num.csv")
    val input2 = sc.textFile(args(1))
    //val input2 = sc.textFile("./data/video_small_testing_num.csv")

    val h1 = input1.first()
    val h2 = input2.first()

    val d1 = input1.filter(x => x!= h1).map(x => x.split(",")).map{x => ((x(0), x(1)), x(2))}.distinct()
    val d2 = input2.filter(x => x!= h2).map(x => x.split(",")).map{x => ((x(0), x(1)), x(2))}.distinct()


    val ratings = d1.subtractByKey(d2).map{case ((user, item), rate) => (user, item, rate)}.map{ case (user, item, rate) =>  Rating(user.toInt, item.toInt, rate.toDouble)}
    val test = d2.map{case ((user, item), rate) => (user, item, rate)}.map{ case (user, item, rate) =>  Rating(user.toInt, item.toInt, rate.toDouble)}

    // Build the recommendation model using ALS
    val rank = 5
    val numIterations = 10
    val model = ALS.train(ratings, rank, numIterations, 0.8 , 1 ,4)


    // Evaluate the model on rating data
    val usersProducts = test.map { case Rating(user, product, rate) =>
      (user, product)
    }

    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

    val pre_max = predictions.map(x => x._2).max
    val pre_min = predictions.map(x => x._2).min

    val predictions_n = predictions.map(x => (x._1, 4*(x._2-pre_min)/(pre_max-pre_min)+1))

    val ratesAndPreds = test.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions_n)

    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()


    val diff = ratesAndPreds.map { case ((user, product), (r1, r2)) => ((user, product), (r1 - r2).abs)}

    val r1 = diff.filter(x => (x._2 >= 0) && (x._2 < 1)).collect().length
    val r2 = diff.filter(x => (x._2 >= 1) && (x._2 < 2)).collect().length
    val r3 = diff.filter(x => (x._2 >= 2) && (x._2 < 3)).collect().length
    val r4 = diff.filter(x => (x._2 >= 3) && (x._2 < 4)).collect().length
    val r5 = diff.filter(x => (x._2 >= 4)).collect().length

    println(s">=0 and <1 : $r1")
    println(s">=1 and <2 : $r2")
    println(s">=2 and <3 : $r3")
    println(s">=3 and <4 : $r4")
    println(s">=4 : $r5")


    val RMSE = pow(MSE, 0.5)

    println(s"Root Mean Squared Error = $RMSE")

    val end_time = System.currentTimeMillis()
    println("===================Count Complete, Total Time: " + (end_time - start_time) / 1000 + " secs====================")


    val filename = args(2)
    //val filename = "./data/Shiwei_Huang_ModelBasedCF.txt"

    val file = new File(filename)

    val out = new PrintWriter(file)

    val fout = predictions_n.sortBy(x =>  (x._1._1, x._1._2)).collect()

    for( i <- 0 to fout.size-1){
      out.write(fout(i)._1._1+","+fout(i)._1._2+","+fout(i)._2+"\n")
    }

    out.close()

    sc.stop()


  }

}
