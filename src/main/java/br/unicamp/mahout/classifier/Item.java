package br.unicamp.mahout.classifier;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

public class Item {

	public static final int NUMBER_OF_ARGUMENTS = 5;
	
	private Vector vector;
	private int actual;
	
	public Item(String values[]) {
		vector = new DenseVector(NUMBER_OF_ARGUMENTS);
		ConstantValueEncoder intercept = new ConstantValueEncoder("intercept");
		StaticWordValueEncoder encoder = new StaticWordValueEncoder("feature");
		
		intercept.addToVector("1", vector);
		
		vector.set(0, Double.valueOf(values[1]));
		vector.set(1, Double.valueOf(values[2]));
		vector.set(2, Double.valueOf(values[3]));
		
		encoder.addToVector(values[0], vector);
		
		actual = Integer.valueOf(values[4]);
	}

	public Vector getVector() {
		return vector;
	}

	public int getActual() {
		return actual;
	}
	
}