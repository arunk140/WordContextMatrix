package liol;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * A special inner class of the MainRunner class.
 * It encapsulates the input object so that other inputs can be used in its place such as a stream
 * input object.
 */
public class InputObject {
	BufferedReader reader;
	
	public InputObject(String fileName) {
		try {
			reader = new BufferedReader(new FileReader(fileName));
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}
	
	public String getNextInstance() {
		try {
			return reader.readLine();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return "mainrunner-error";
	}
}
