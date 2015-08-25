package br.unicamp.mahout.cluster;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import br.unicamp.mahout.utils.MyFileReader;

public class ClusterExample {

	private static final String POINTS_PATH = "/home/everton.gago/workspace/mahout-exemplos/points";
    private static final String CLUSTERS_PATH = "/home/everton.gago/workspace/mahout-exemplos/clusters";
    private static final String OUTPUT_PATH = "/home/everton.gago/workspace/mahout-exemplos/output";
    private static final String CLUSTERED_POINTS_DIR = "/home/everton.gago/workspace/mahout-exemplos/output/clusteredPoints";
    private static final String INPUT_PATH = "/home/everton.gago/workspace/mahout-exemplos/classify/data-cluster.csv";

    private final int NUMBER_OF_CLUSTERS = 4;
	
	public void start() throws IOException, ClassNotFoundException, InterruptedException {
		Configuration configuration = new Configuration();
		
		final List<Vector> points = this.readLines();
		
		File pointsDir = new File(POINTS_PATH);
		if (!pointsDir.exists())
			pointsDir.mkdir();

		this.writePoints(configuration, points);
		
		this.initialClustersPositions(configuration, points);
		
		this.runKmeans(configuration);
		
		this.printClusters(configuration);
	}

	private void writePoints(Configuration configuration, final List<Vector> points) throws IOException {
		Path pointsPath = new Path(POINTS_PATH + "/pointsFile");
		SequenceFile.Writer pointsWriter = SequenceFile.createWriter(
				configuration,
				SequenceFile.Writer.file(pointsPath),
				SequenceFile.Writer.keyClass(IntWritable.class),
				SequenceFile.Writer.valueClass(VectorWritable.class));
		int recNum = 0;
		VectorWritable vectorWritable = new VectorWritable();
		for (Vector point : points) {
			vectorWritable.set(point);
			pointsWriter.append(new IntWritable(recNum++), vectorWritable);
		}
		pointsWriter.close();
	}

	private void initialClustersPositions(Configuration configuration, final List<Vector> points) throws IOException {
		Path centersPath = new Path(CLUSTERS_PATH + "/part-00000");
		SequenceFile.Writer centersWriter = SequenceFile.createWriter(
				configuration,
				SequenceFile.Writer.file(centersPath),
				SequenceFile.Writer.keyClass(Text.class),
				SequenceFile.Writer.valueClass(Kluster.class));
		for (int i = 0; i < NUMBER_OF_CLUSTERS; i++) {
			Vector point = points.get(i);
			Kluster cluster = new Kluster(point, i, new EuclideanDistanceMeasure());
			centersWriter.append(new Text(cluster.getIdentifier()), cluster);
		}
		centersWriter.close();
	}

	private void runKmeans(Configuration configuration) throws IOException, InterruptedException, ClassNotFoundException {
		Path inputPath = new Path(POINTS_PATH);
		Path clusterPath = new Path(CLUSTERS_PATH);
		Path outputPath = new Path(OUTPUT_PATH);
		HadoopUtil.delete(configuration, outputPath);
		
		// TODO
		KMeansDriver.run(configuration, inputPath, clusterPath, outputPath, 0.001, 10, true, 0, false);
		// FuzzyKMeansDriver.run(configuration, inputPath, clusterPath, outputPath, 0.001, 10, 1.1F, true, true, 0.001, false);
	}

	private void printClusters(Configuration configuration) throws IOException {
		Path resultFile = new Path(CLUSTERED_POINTS_DIR + "/part-m-00000");
		SequenceFile.Reader reader = new SequenceFile.Reader(configuration, SequenceFile.Reader.file(resultFile));
		IntWritable key = new IntWritable();
		WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
		while (reader.next(key, value))
			System.out.printf("item %s: cluster: %s\n", value.toString(), key.toString());
		reader.close();
	}

	private List<Vector> readLines() throws FileNotFoundException {
		List<String[]> lines = MyFileReader.loadData(INPUT_PATH);
		final List<Vector> points = new ArrayList<Vector>();
		for (String line[] : lines) {
			Vector vec = new RandomAccessSparseVector(line.length);
			
			double[] linesDouble = new double[line.length];
			for (int j = 0; j < linesDouble.length; j++) {
				linesDouble[j] = Double.parseDouble(line[j]);
			}
			
			vec.assign(linesDouble);
			points.add(vec);
		}
		return points;
	}
	
	public static void main(String[] args) throws ClassNotFoundException, IOException, InterruptedException {
		ClusterExample cluster = new ClusterExample();
		cluster.start();
	}
	
}