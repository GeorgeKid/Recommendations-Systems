import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}

import scala.math.{min, pow}

object ItemBasedCF {

  def isCoprime(x:Int, y:Int):Boolean = {
    if (x == 1 || y == 1){
      return true
    }

    var a = x
    var b = y

    var t = a % b
    while(t!=0){
      a = b
      b = t
      t = a % b

    }

    if (b > 1) return false
    else return true

  }

  def main (args: Array[String]): Unit = {

    var conf = new SparkConf()
    conf.setAppName("ItemBasedCF")
    conf.setMaster("local[4]")

    val sc = new SparkContext (conf)
    val input1 = sc.textFile(args(0))
    //val input1 = sc.textFile("./data/video_small_num.csv")
    val input2 = sc.textFile(args(1))
    //val input2 = sc.textFile("./data/video_small_testing_num.csv")

    val h1 = input1.first()
    val h2 = input2.first()

    /////////////////// Calculate the Jaccard Similarity Pairs //////////////////

    //distinct users
    val users = input1.filter(x => x!= h1)
      .map(x => x.split(",")).map(x => x(0)).distinct()

    val users_length = users.collect().length

    //user => (user, id)
    val user_index = users.zipWithIndex()

    //println(user_index.collect()(1))


    //character matrix
    val user_item = input1.filter(x => x!= h1)
      .map(x => x.split(",")).map(x => (x(0), x(1)))

    //(item, (u1, u2, u3...))
    val item_us = user_index.join(user_item).map{ case(user, (user_id, item))=> (item, user_id)}.groupByKey()

    //println(item_us.collect().length) OK

    //signature matrix
    val b = 100
    val r = 3
    val n = b * r

    val b_value = scala.collection.mutable.Set[Int]()
    val a_value = scala.collection.mutable.Set[Int]()

    while (b_value.size != n){
      b_value += (new util.Random).nextInt(Int.MaxValue)
    }

    while (a_value.size != n){
      val av = (new util.Random).nextInt(Int.MaxValue)
      if(isCoprime(av, users_length)){
        a_value += av
      }
    }

    val as = a_value.toList
    val bs = b_value.toList

    //println(as.length) OK

    //build the sig_matrix
    val sig_matrix = item_us.mapValues( x => {
      var h_list = scala.collection.mutable.ListBuffer[Int]()
      for (k <- 0 to n-1 ){
        var h_value = users_length
        for (user_id <- x) {
          h_value = min(h_value, (as(k) * user_id.toInt + bs(k)) % users_length)
        }
        h_list.append(h_value)
      }
      h_list.toList
    })

    //divide band (item, ((h1, h2, .. hr), band_id))
    val sig_band = sig_matrix.flatMapValues(x => x.grouped(r).zipWithIndex)
      .map{case (item, (list_h, band_id)) => ((list_h, band_id), item)}

    val candidates = sig_band.groupByKey().map(x => x._2.toSet).filter(x => x.size >= 2)
      .flatMap(x => x.subsets(2))
      .map(x => if (x.toList(0).toInt < x.toList(1).toInt) (x.toList(0), x.toList(1)) else (x.toList(1), x.toList(0))).distinct()

    //add item1_users (item1, (item2, item1_users)) => (item2, (item1, item1_users))
    val s1 = candidates.join(item_us).map{ case (item1, (item2, item1_users)) => (item2, (item1, item1_users))}
    //add item2_users// (item2, ((item1, item1_users), item2_users)) => ((item1, item2), (item1_users, item2_users))
    val s2 = s1.join(item_us).map{case (item2, ((item1, item1_users), item2_users)) => ((item1, item2), (item1_users.toSet, item2_users.toSet))}
      .map{case ((item1, item2), (u1, u2)) => ((item1, item2), u1.intersect(u2).size.toDouble/u1.union(u2).size)}

    val true_can = s2.filter(x => x._2 >= 0.5).map(x =>(x._2, x._1)).flatMap(x => Set((x._1, x._2), (x._1, x._2.swap))).map(x => ((x._2._1.toInt, x._2._2.toInt), x._1))

    //println("true_can:" + true_can.collect().length)

    ////////////////// Calculate the Predection value in training data ///////////////

    //record start-time
    val start_time = System.currentTimeMillis()


    val d1 = input1.filter(x => x!= h1).map(x => x.split(",")).map{x => ((x(0).toInt, x(1).toInt), x(2).toDouble)}
    //testing data, user_id, item_id, rating
    val d2 = input2.filter(x => x!= h2).map(x => x.split(",")).map{x => ((x(0).toInt, x(1).toInt), x(2).toDouble)}

    //training data, user_id, item_id, rating
    val tran_data = d1.subtractByKey(d2).map(x => (x._1._1.toInt, x._1._2.toInt, x._2.toDouble))


    //(user, (item, rating))
    val itemi_user = tran_data.map{ case(user, itemi, ratingi) => (user, (itemi, ratingi))}

    val itemj_user = tran_data.map{ case(user, itemj, ratingj) => (user, (itemj, ratingj))}

    //(user, ((itemi, ratingi), (itemj, ratingj))) => ((itemi,itemj), (ratingi, ratingj))
    val user_item_i_j = itemi_user.join(itemj_user).filter( x => x._2._1._1 != x._2._2._1)
      //.map{case (user, ((itemi, ratingi), (itemj, ratingj))) => ((itemi,itemj), (ratingi, ratingj, 1))}.reduceByKey((x, y) => (x._1+y._1, x._2+y._2, x._3+y._3))

    val user_item_ave = user_item_i_j.map{case (user, ((itemi, ratingi), (itemj, ratingj))) => ((itemi,itemj), (ratingi, ratingj, 1))}.reduceByKey((x, y) => (x._1+y._1, x._2+y._2, x._3+y._3))
      .map{case ((itemi, itemj), (ratingi_sum, ratingj_sum, co_user_size))=> ((itemi, itemj), (ratingi_sum/co_user_size, ratingj_sum/co_user_size))}

   val user_item_com = user_item_i_j.map{case (user, ((itemi, ratingi), (itemj, ratingj))) => ((itemi,itemj), (ratingi, ratingj))}
    .join(user_item_ave).map{case ((itemi, itemj), ((ratingi, ratingj), (avei, avej)))=> ((itemi, itemj), ratingi - avei, ratingj- avej)}

    //    val s_items = itemi_user.groupByKey().map(x => x._2.size)
//
//    println("s_items:" + s_items.min) 同一个user下item最少6个

    //println("user_item_i_j size: "+user_item_i_j.collect().size)

    //((itemi, itemj, ratingi, ratingj), (ratingi_sum, ratingj_sum, co_user_size)) => ((itemi, itemj, ratingi, ratingj), ratingi_sum/co_user_size, ratingj_sum/co_user_size))
//    val user_item_com = user_item_i_j.map{ case((itemi, itemj,ratingi, ratingj), (ratingi_sum, ratingj_sum, co_user_size))=> ((itemi, itemj, ratingi, ratingj), ratingi_sum/co_user_size, ratingj_sum/co_user_size)}
//      .map{ case ((itemi, itemj, ratingi, ratingj), ai, aj) => ((itemi, itemj), ratingi -ai, ratingj- aj)}

    //Cal W
    val W_ii_ij = user_item_com.map{ case ((itemi, itemj), di, dj) => ((itemi, itemj), (di*dj, di*di, dj*dj))}.reduceByKey((x, y) => (x._1+y._1, x._2+y._2, x._3+y._3))
      .mapValues(x => if (x._2*x._3 != 0) x._1/pow(x._2*x._3, 0.5) else 0.0)

    //println("Wii_ij_max: "+ W_ii_ij.map(x => x._2).max)
    //////////////////////Change W based on Jaccard similarity, W = 0.0, if they are not similar ////////////////

    //println("W总："+W_ii_ij.collect().length)  689544
    //not similar, w = 0.0
    val W_1 = W_ii_ij.subtractByKey(true_can).map(x => (x._1, 0.0))

    //println("W不："+ W_1.collect().length) 587356

    //similar, not change , plus the not simailar one
    val W_final = W_ii_ij.subtractByKey(W_1) .++ (W_1)

    //println("W_final_size: "+ W_final.collect().size)



    //test data -> (user_t, item_t)
    val user_t_item_t = d2.map{ case((user_t, item_t), rating_t) => (user_t, item_t)}
    //tran date -> (user, (item, rating))
    val user_i = tran_data.map{ case(user, item, rating) => (user, (item, rating))}

    //(user_t, (item_t, (item, rating))) => (user_t, (item_t, item, rating))
    val test_join_tran = user_t_item_t.join(user_i).map{case (user_t, (item_t, (item, rating))) => (user_t, (item_t, item, rating))}

//    println("test:" + user_t_item_t.collect().length) 7700
//    println("train: " + user_i.collect().size) 38558
//    println("test_join"+test_join_tran.collect().size) 172977
//
//    val size_join = test_join_tran.groupByKey().map(x => x._2.size).min   6
//
   // println("test_join_tran size: "+ test_join_tran.collect().length )

    // add W_final ((item, item_t), ((user_t, rating), w)) => (user_t, (w, rating, item_t, item))
    val P_com = test_join_tran.map{ case (user_t, (item_t, item, rating)) => ((item, item_t), (user_t, rating))}.join(W_final)
      .map{ case ((item, item_t), ((user_t, rating), w)) => (user_t, (w, rating, item_t, item))}

    //println("w_max: "+ P_com.map(x => x._2._1).max)

   // println("P_com_size: "+ P_com.collect().size)

//    val size = P_com.groupByKey().map(x => x._2.size).min
//    println("size: " + size)

    // Set N = 5, 因为一个user下最少1个items，小于5的全部相加
    val N = 5
    val P = P_com.groupByKey().map(x => (x._1, x._2.toList.sortBy(x => -x._1))).map{ x => {
      var P_up = 0.0
      var P_down = 0.0
      var item_t = 0

      var l = 0
      if (x._2.size >= N) l = N
      else l = x._2.size

      for( i<- 0 to l-1 ){
        item_t = x._2(i)._3
        P_up += x._2(i)._1 * x._2(i)._2
        P_down += x._2(i)._1.abs

      }
      if (P_down != 0 ) ((x._1,item_t), P_up/ P_down)
      else ((x._1,item_t), 0.0)
    }}.filter(x => x._2 != 0.0)

    println(P.collect().size)

    //(item, ave_rating)
    val item_ave_rating = tran_data.map{ case(user, item, rating) => (item, (rating, 1))}.reduceByKey((x, y) => (x._1+y._1, x._2+y._2)).mapValues(x => x._1/x._2)

    //rest user_t
    val P_r = d2.subtractByKey(P).map{ case ((user_t, item_t), rating) => (item_t, user_t)}.join(item_ave_rating)
      .map{ case (item_t, (user_t, ave_rating_it)) => ((user_t, item_t), 4.0)}

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
    //val filename = "./data/*Shiwei_Huang_ItemBasedCF.txt"

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
