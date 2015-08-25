package br.unicamp.mahout.classifier;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.Vector;

import br.unicamp.mahout.utils.MyFileReader;

public class ClassifyExample {

	private static final String INPUT_PATH = "/home/everton.gago/workspace/mahout-exemplos/classify/data-training.csv";
	private static final int EPOCHS = 1000;

	public void start() throws FileNotFoundException {
		List<Item> inputs = this.readLines();
		OnlineLogisticRegression classifier = trainClassifier(inputs);
		
		// 2 = not classified
		String[] inputTest = new String[] { "ANOMALO", "24", "595", "43", "2" };
		this.testClassifier(classifier, inputTest);
		
		classifier.close();
	}

	private void testClassifier(OnlineLogisticRegression classifier, String[] inputTest) {
		Item item = new Item(inputTest);
		Vector result = classifier.classifyFull(item.getVector());
		System.out.println("---------- Testing ----------");
		System.out.format("Probability of not fraud (0) = %.3f\n", result.get(0));
		System.out.format("Probability of fraud (1) = %.3f\n", result.get(1));
	}

	private OnlineLogisticRegression trainClassifier(List<Item> inputs) {
		OnlineLogisticRegression classifier = new OnlineLogisticRegression(2, Item.NUMBER_OF_ARGUMENTS, new L1());
		for (int i = 0; i < EPOCHS; i++) {
			for (Item item : inputs) {
				classifier.train(item.getActual(), item.getVector());
			}
			
			// accuracy.
			/*if (i % 50 == 0) {
				Auc auc = new Auc(0.5);
				for (Item item : inputs) {
					auc.add(item.getActual(), classifier.classifyScalar(item.getVector()));
					
					System.out.format("Epoch: %2d, Learning rate: %2.4f, Accuracy: %2.4f\n", 
							i, classifier.currentLearningRate(), auc.auc());
				}
				System.out.println("");
			}*/
		}
		return classifier;
	}

	private List<Item> readLines() throws FileNotFoundException {
		List<String[]> lines = MyFileReader.loadData(INPUT_PATH);
		List<Item> inputs = new ArrayList<Item>();
		for (String[] value : lines) {
			Item item = new Item(value);
			inputs.add(item);
		}
		return inputs;
	}
	
	public static void main(String[] args) throws FileNotFoundException {
		ClassifyExample example = new ClassifyExample();
		example.start();
	}

}