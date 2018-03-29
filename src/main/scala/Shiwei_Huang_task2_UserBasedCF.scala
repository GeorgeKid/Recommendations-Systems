import java.io.{File, PrintWriter}

import scala.math._
import org.apache.spark.{SparkConf, SparkContext}

object UserBasedCF {

  def main (args: Array[String]): Unit = {
    var conf = new SparkConf()
    conf.setAppName("UserBasedCF")
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

    val d1 = input1.filter(x => x!= h1).map(x => x.split(",")).map{x => ((x(0).toInt, x(1).toInt), x(2).toDouble)}
    //testing data, user_id, item_id, rating
    val d2 = input2.filter(x => x!= h2).map(x => x.split(",")).map{x => ((x(0).toInt, x(1).toInt), x(2).toDouble)}

    //training data, user_id, item_id, rating
    val tran_data = d1.subtractByKey(d2).map(x => (x._1._1.toInt, x._1._2.toInt, x._2.toDouble))

    //(user, ave_rating)
    val user_ave_rating = tran_data.map{ case(user, item, rating) => (user, (rating, 1))}.reduceByKey((x, y) => (x._1+y._1, x._2+y._2)).mapValues(x => x._1/x._2)

    //(user, (item, rating))
    val useri_item = tran_data.map{ case(useri, item, ratingi) => (useri, (item, ratingi))}

    val useri_item_ave = user_ave_rating.join(useri_item).map{ case (useri, (ave_ratingi,(item, ratingi))) => (item, (useri, ratingi, ave_ratingi))}

    val userj_item = tran_data.map{ case(userj, item, ratingj) => (userj, (item, ratingj))}

    val userj_item_ave = user_ave_rating.join(userj_item).map{ case (userj, (ave_ratingj,(item, ratingj))) => (item, (userj, ratingj, ave_ratingj))}

    val item_useri_userj = useri_item_ave.join(userj_item_ave).filter(x => x._2._1._1 != x._2._2._1)//eliminate repeat

    //((useri, userj), (ratingi-ave_ratingi, ratingj-ave_ratingj))
    val W_com = item_useri_userj.map{ case (item, ((useri, ratingi, ave_ratingi), (userj, ratingj, ave_ratingj))) => ((useri, userj), (ratingi-ave_ratingi, ratingj-ave_ratingj))}

    //W cal
    val W_ui_uj = W_com.map{ case ((useri, userj), (di, dj)) => ((useri, userj), (di*dj, di*di, dj*dj))}.reduceByKey((x, y) => (x._1+y._1, x._2+y._2, x._3+y._3))
      .mapValues(x => if (x._2*x._3 != 0) x._1/pow(x._2*x._3, 0.5) else 0.0)

    //println(W_ui_uj.collect().length) Ok

    //println(W)

    //test data -> (item_t, user_t)
    val item_t_user_t = d2.map{ case((user_t, item_t), rating_t) => (item_t, user_t)}
    //tran date -> (item, (user, rating))
    val item_user = tran_data.map{ case(user, item, rating) => (item, (user, rating))}

    //(item, (user_t, (user, rating))) => (user_t, (item_t, user, rating))
    val test_join_tran = item_t_user_t.join(item_user).map{ case (item_t, (user_t, (user, rating))) => (user_t, (item_t, user, rating))}

    //add ave_rating_ut : (user_t, ((item_t, user, rating), ava_rating_ut)) => (user, (item_t, user_t, rating, ava_rating_ut))
    val at = test_join_tran.join(user_ave_rating).map{ case (user_t, ((item_t, user, rating), ave_rating_ut)) => (user, (item_t, user_t, rating, ave_rating_ut))}

    //add ave_rating_u : (user, ((item_t, user_t, rating, ave_rating_ut), ave_rating_u) => ((user_t, user), (item_t, rating, ave_rating_u, ave_rating_ut))
    val ua = at.join(user_ave_rating).map{ case (user, ((item_t, user_t, rating, ave_rating_ut), ave_rating_u))=> ((user_t, user), (item_t, rating, ave_rating_ut, ave_rating_u))}


    println("ua size: "+ua.collect().length)
    //add W_ut_u: ((user_t, user), ((item_t, rating, ave_rating_u, ava_rating_ut), W_ut_u)) => ((user_t,item_t), ((rating-ave_rating_u)*w_ut_u, w_ut_u.abs, ave_rating_ut))
    val P_com = ua.join(W_ui_uj).map{ case ((user_t, user), ((item_t, rating, ave_rating_ut, ave_rating_u), w_ut_u)) => ((user_t,item_t), ((rating-ave_rating_u)*w_ut_u, w_ut_u.abs, ave_rating_ut))}
    println("P_com size:" +P_com.collect().length)
    //cal P
    val P = P_com.reduceByKey((x, y) => (x._1+y._1, x._2+y._2, x._3)).mapValues(x => if (x._2 != 0) x._3 + x._1/x._2 else x._3)

    //rest user_t
    val P_r = d2.subtractByKey(P).map{ case ((user_t, item_t), rating) => (user_t, item_t)}.join(user_ave_rating)
      .map{ case (user_t, (item_t, ave_rating_ut)) => ((user_t, item_t), ave_rating_ut)}

   // val P_r = d2.subtractByKey(P).map{ case ((user_t, item_t), rating) => ((user_t, item_t), 4.0)}

    val predictions = P .++(P_r)

    val p_max = predictions.map(x => x._2).max
    val p_min = predictions.map(x => x._2).min

    //normalize the predict result to [1, 5]
    //val predictions_n = predictions.map(x => (x._1, 4*(x._2-p_min)/(p_max-p_min)+1))

    //println(predictions.collect().length)

    //true and pre values
    val trueAndPre = d2.join(predictions)


    val MSE = trueAndPre.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()

    val diff = trueAndPre.map { case ((user, product), (r1, r2)) => ((user, product), (r1 - r2).abs)}

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
    //val filename = "./data/Shiwei_Huang_UserBasedCF.txt"

    val file = new File(filename)

    val out = new PrintWriter(file)

    val fout = predictions.sortBy(x =>  (x._1._1, x._1._2)).collect()

    for( i <- 0 to fout.size-1){
      out.write(fout(i)._1._1+","+fout(i)._1._2+","+fout(i)._2+"\n")
    }

    out.close()

    sc.stop()


  }

}
