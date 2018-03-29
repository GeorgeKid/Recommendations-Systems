import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}

import scala.math._

object JaccardLSH{

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
    conf.setAppName("JaccardLSH")
    conf.setMaster("local[4]")

    val sc = new SparkContext (conf)

    val input = sc.textFile(args(0))
    //val input = sc.textFile("./data/video_small_num.csv")

    val header = input.first()

    //distinct users
    val users = input.filter(x => x!= header)
      .map(x => x.split(",")).map(x => x(0)).distinct()

    val users_length = users.collect().length

    //user => (user, id)
    val user_index = users.zipWithIndex()

    //println(user_index.collect()(1))


    //character matrix
    val user_item = input.filter(x => x!= header)
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

    val true_can = s2.filter(x => x._2 >= 0.5)

    //println("true_can:"+true_can.collect().size)


    //read the truth Jaccard pairs
//    val in = sc.textFile(args(1))
//    //val in = sc.textFile("./data/video_small_ground_truth_jaccard.csv")
//
//    val head = in.first()
//
//    val truepairs = in.filter(x => x!= head)
//      .map(x => x.split(",")).map(x => (x(0), x(1))).collect()
//
//    val canpairs = true_can.map(x => x._1).collect()
//
//    //true positive
//    val tp = truepairs.toSet & canpairs.toSet
//    //false positive
//    val fp = canpairs.toSet -- truepairs.toSet
//    // false negative
//    val fn = truepairs.toSet -- canpairs.toSet
//
//    println("P:"+tp.size.toDouble/tp.size.toDouble)
//    println("R:"+tp.size.toDouble/(tp.size+fn.size).toDouble)


    val filename = args(1)
    //val filename = "./data/Shiwei_Huang_SimilarProducts_Jaccard.txt"

    val file = new File(filename)

    val out = new PrintWriter(file)

    val fout = true_can.collect().sortBy(x => (Integer.parseInt(x._1._1), Integer.parseInt(x._1._2)))

    for( i <- 0 to fout.size-1){
      out.write(fout(i)._1._1+","+fout(i)._1._2+","+ fout(i)._2.toString+"\n")
    }

    out.close()

  }

}
