package main.java.liol;

import moa.core.TimingUtils;

import java.util.ArrayList;

/**
 * <h1>Build a word-context matrix before running a learner on the sparse word vectors!</h1>
 * The MainRunner program acts as a governor program taking input, building a matrix out of it and
 * then performing classification on the representations of the words.
 *
 * @author Tristan Anderson
 * @version 1.0
 * @since 2018-07-13
 */
public class MainRunner {
  
  /**
   * This method will call the filter software and then feeds the output into an SGD classifier
   *
   * @param args Takes as arugments the input file name, output file name, vocabulary size, context
   *             size and window size.
   * @return nothing
   * @exception IllegalArgumentException on arguments error.
   * @see IllegalArgumentException
   */
  public static void main(String[] args) {
    System.err.println(System.getProperty("user.dir"));
    try {
      if (args.length != 6) {
        
        System.err.println("Usage: [SeedLexicon][InputFileName][OutputFileName]" +
            "[VocabSize][ContextSize][WindowSize]");
        throw (new IllegalArgumentException());
      } else { // Success so now we do some pre-processing before feeding it into the SGD learner
        
        MainRunner runner = new MainRunner();
        
        ArrayList<Integer> params = new ArrayList<>();
        InputObject seedLex = new InputObject(args[0]);
        InputObject inStream = new InputObject(args[1]);
        
        for (int i = 3; i < 6; i++) {
          if (TryParse(args[i])) {
            params.add(Integer.parseInt(args[i]));
          } else {
            throw new IllegalArgumentException("Please use integers to describe parameters.");
          }
        }
        runner.run(args[2], params, seedLex, inStream);
      }
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
  
  /**
   * This method attempts to parse a string into an integer and controls an exception if it fails.
   * @param s Some string to be converted to an integer but an error is expected.
   * @return boolean A boolean success value of the conversion.
   */
  public static boolean TryParse(String s) {
    try {
      Integer.parseInt(s);
      return true;
    } catch (NumberFormatException ex) {
      return false;
    }
  }
  
  /**
   * Runs the filter software and eventually the classifiers.
   * @param outName The name of the output file
   * @param params A list of the parameters for the word context matrix
   * @param seedLex The lexicon of known words and their polarities.
   */
  private void run(String outName, ArrayList<Integer> params, InputObject seedLex,
                   InputObject inputStream) {
    
    boolean preceiseCPUTiming = TimingUtils.enablePreciseTiming();
    long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
    
    // Read in the lexicon and give it to the trainer.
    Trainer trainer = new Trainer(evaluateStartTime);
    trainer.Initialize(seedLex);
    
    WordContextMatrix wcm = new WordContextMatrix(params.get(0), params.get(1),
        params.get(2), inputStream, outName, trainer);
    wcm.buildMatrix();
  }
}
