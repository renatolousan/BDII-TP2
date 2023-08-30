from pyspark import SparkConf, SparkContext

def compute_contribs(pair):
    _, (links, rank) = pair
    num_links = len(links)
    for dest in links:
        yield (dest, rank / num_links)

def main():
    conf = SparkConf().setAppName("PageRank")
    sc = SparkContext(conf=conf)

    links = sc.textFile("gs://$BUCKET_NAME/input_data").map(lambda line: tuple(line.split())).groupByKey().cache()
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))

    NUM_ITERATIONS = 10
    for _ in range(NUM_ITERATIONS):
        contribs = links.join(ranks).flatMap(compute_contribs)
        ranks = contribs.reduceByKey(lambda x, y: x + y).mapValues(lambda rank: 0.15 + 0.85 * rank)

    ranks.saveAsTextFile("gs://$BUCKET_NAME/output_data")

if __name__ == "__main__":
    main()
