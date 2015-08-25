package br.unicamp.mahout.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class MyFileReader {

	public static List<String[]> loadData(String path) throws FileNotFoundException {
		List<String[]> data = new ArrayList<String[]>();
		
		Scanner scan = new Scanner(new FileInputStream(new File(path)));
		scan.nextLine();
		while (scan.hasNextLine()) {
			data.add(scan.nextLine().split(","));
		}
		scan.close();
		return data;
	}
	
}