package com.soundcloud.lsh
import java.io.{File, PrintWriter}

import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

object CosineLSH {

  def main(args: Array[String]) {

    // init spark context
    val numPartitions = 8
    //val input = "data/video_small_num.csv"
    val conf = new SparkConf()
      .setAppName("LSH-Cosine")
      .setMaster("local[4]")
    val storageLevel = StorageLevel.MEMORY_AND_DISK
    val sc = new SparkContext(conf)

    //read in an example data set of word embeddings
//    val data = sc.textFile(input, numPartitions).map {
//      line =>
//        val split = line.split(" ")
//        val word = split.head
//        val features = split.tail.map(_.toDouble)
//        (word, features)
//    }

    val input = sc.textFile("./data/video_small_num.csv", numPartitions)

    val header = input.first()

    val data = input.filter(x => x!= header)
      .map(x => x.split(",")).map(x => (x(1).toInt, x(0).toInt))
      .groupByKey()

    val colNum = data.map({ case (x, y) => (x, y.max)}).map({case (x, y) => y}).max + 1

    // create an unique id for each word by zipping with the RDD index
    val indexed = data.zipWithIndex.persist(storageLevel)

    // create indexed row matrix where every row represents one word
    val rows = indexed.map {
      case ((word, features), index) =>
        val feaArr = new SparseVector(colNum, features.toArray, Array.fill(features.size)(1.0))
        IndexedRow(index, Vectors.dense(feaArr.toArray))
    }

//    val rows = data.map({
//      case (x, y) => IndexedRow(x, new SparseVector(colNum, y.toArray, Array.fill(y.size)(1.0)))
//    }).cache()

    // store index for later re-mapping (index to word)
    val index = indexed.map {
      case ((word, features), index) =>
        (index, word)
    }.persist(storageLevel)

    // create an input matrix from all rows and run lsh on it
    val matrix = new IndexedRowMatrix(rows)
    val lsh = new Lsh(
      minCosineSimilarity = 0.5,
      dimensions = 20,
      numNeighbours = 200,
      numPermutations = 20,
      partitions = numPartitions,
      storageLevel = storageLevel
    )

    val similarityMatrix = lsh.join(matrix)

    // remap both ids back to words
    val remapFirst = similarityMatrix.entries.keyBy(_.i).join(index).values
    val remapSecond = remapFirst.keyBy { case (entry, word1) => entry.j }.join(index).values.map {
      case ((entry, word1), word2) =>
        (word1, word2, entry.value)
    }

    // group by neighbours to get a list of similar words and then take top k
//    val result = remapSecond.groupBy(_._1).map {
//      case (word1, similarWords) =>
//        // sort by score desc. and take top 10 entries
//        val similar = similarWords.toSeq.sortBy(-1 * _._3).take(10).map(_._2).mkString(",")
//        s"$word1 --> $similar"
//    }

    val result = remapSecond.map({case(x, y, z) => if(x > y)  (y, x) else (x, y)}).collect().toSet
    println(result.size)

    //read the truth cosine pairs
    val in = sc.textFile("./data/video_small_ground_truth_cosine.csv")

    val head = in.first()

    val truepairs = in.filter(x => x!= head)
      .map(x => x.split(",")).map(x => (x(0).toInt, x(1).toInt)).collect().toSet
    println(truepairs.size)
    val tp = truepairs & result
    //false positive
    val fp = result -- truepairs
    //false negative
    val fn = truepairs -- result

    println(tp.size)
    println(fn.size)
    println("P:"+tp.size.toDouble/result.size.toDouble)
    println("R:"+tp.size.toDouble/(tp.size+fn.size).toDouble)

//    // print out the results for the first 10 words
//    result.take(20).foreach(println)


    val filename = "./data/Shiwei_Huang_SimilarProducts_Cosine.txt"

    val file = new File(filename)

    val out = new PrintWriter(file)

    val fout2 = remapSecond.map({case(x, y, z) => if(x > y)  (y, x, z) else (x, y, z)}).sortBy(x =>  (x._1, x._2)).collect()

    for( i <- 0 to fout2.size-1){
      out.write(fout2(i)._1+","+fout2(i)._2+","+fout2(i)._3+"\n")
    }

    out.close()

    sc.stop()

  }




}
