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
   * @param args Takes as arugments the input file name, vocabulary size, context
   *             size and window size.
   * @return nothing
   * @exception IllegalArgumentException on arguments error.
   * @see IllegalArgumentException
   */
  public static void main(String[] args) {
    System.err.println(System.getProperty("user.dir"));
    try {
      if (args.length != 7) {
        
        System.err.println("Usage: [SeedLexicon][InputFileName]" +
            "[VocabSize][ContextSize][WindowSize][SketchingMethod][WeightingMethod]");
        throw (new IllegalArgumentException());
      } else { // Success so now we do some pre-processing before feeding it into the SGD learner
        
        MainRunner runner = new MainRunner();
        
        ArrayList<Integer> params = new ArrayList<>();
        InputObject seedLex = new InputObject(args[0]);
        InputObject inStream = new InputObject(args[1]);
        
        for (int i = 2; i < 5; i++) {
          if (TryParse(args[i])) {
            params.add(Integer.parseInt(args[i]));
          } else {
            throw new IllegalArgumentException("Please use integers to describe parameters.");
          }
        }
        runner.run(seedLex, inStream, params, Integer.parseInt(args[5]), Integer.parseInt(args[6]));
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
   * @param params A list of the parameters for the word context matrix (vocab size, context size,
   *               window size).
   * @param seedLex The lexicon of known words and their polarities.
   * @param sketch The sketching choice
   * @param weight The weighting choice
   */
  private void run(InputObject seedLex, InputObject inputStream, ArrayList<Integer> params,
                   int sketch, int weight) {
    
    boolean preceiseCPUTiming = TimingUtils.enablePreciseTiming();
    long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
    
    // Read in the lexicon and give it to the trainer.
    Trainer trainer = new Trainer(evaluateStartTime);
    trainer.Initialize(seedLex);
    //System.err.println("Vocab size: " + params.get(0) + " Context size: " + params.get(1) +
    // " Window size: " + params.get(2));
    
    WordContextMatrix wcm = new WordContextMatrix(params.get(0), params.get(1),
        params.get(2), inputStream, trainer);

    // Set the sketching method
    wcm.setSketchingMethod(sketch);
    
    // Set the weighting method
    wcm.setWeightingMethod(weight);
    
    // Begin
    wcm.buildMatrix();
  }
}
