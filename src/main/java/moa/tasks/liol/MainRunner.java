package moa.tasks.liol;

import com.github.javacliparser.ClassOption;
import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import moa.classifiers.functions.SGD;
import moa.classifiers.Classifier;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.TimingUtils;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.learners.Learner;
import moa.tasks.TaskMonitor;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * <h1>Build a word-context matrix before running a learner on the sparse word vectors!</h1>
 * The MainRunner program acts as a governor program taking input, building a matrix out of it and
 * then performing classification on the representations of the words.
 *
 * @author Tristan Anderson
 * @version 1.0
 * @since 2018-07-13
 */
public class MainRunner extends moa.tasks.ClassificationMainTask {
  @Override
  public String getPurposeString() {
    return "Build a word-context matrix before running a learner on the sparse word vectors for performing classification on the representations of the words";
  }

  private static final long serialVersionUID = 1L;


  public FileOption SeedLexiconTrain = new FileOption("SeedLexiconTrain", 'd',
          "File with the SeedLexicon Train Set", null, "txt", true);
  public FileOption SeedLexiconTest = new FileOption("SeedLexiconTest", 't',
          "File with the SeedLexicon Test Set", null, "txt", true);

  public FileOption InputFileName = new FileOption("InputFileName", 'o',
          "File with the Input Stream", null, "txt", true);

  public IntOption vocabSizeOption = new IntOption("vocabSizeOption", 'v',
          "Max Size of Vocabulary",
          100000, 100, Integer.MAX_VALUE);;
  public IntOption contextSizeOption = new IntOption("contextSizeOption", 'c',
          "Size of Context",
          10000, 100, Integer.MAX_VALUE);;
  public IntOption windowSizeOption = new IntOption("windowSizeOption", 'w',
          "Size of the Window",
          4, 1, 5);

  public FlagOption enableHashing = new FlagOption("enableHashing", 'h', "Enable Hashing");
  public FlagOption enablePPMI = new FlagOption("enablePPMI", 'p', "Use PPMI");

  public IntOption sampleFrequency = new IntOption("sampleFrequency", 'f',
          "Sample Frequency",
          1000, 100, Integer.MAX_VALUE);
//
//  public ClassOption learnerOption = new ClassOption("learner", 'l', "Classifier to train.", Classifier.class,
//          "functions.SGD");


    Trainer trainer;
  static LearningCurve learningCurve;


  
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
      if (args.length != 9) {
        System.err.println("Usage: [SeedLexiconTrain][InputFileName]" +
            "[VocabSize][ContextSize][WindowSize][SketchingMethod][WeightingMethod][SampleFrequency][SeedLexiconTest]");
        System.err.println("Your input: " + Arrays.toString(args));
//        System.err.println("Your input: " + Arrays.toString(args));
        throw (new IllegalArgumentException());
      } else { // Success so now we do some pre-processing before feeding it into the SGD learner
        
        MainRunner runner = new MainRunner();
        
        ArrayList<Integer> params = new ArrayList<>();
        InputObject seedLexTrain = new InputObject(args[0]);
        InputObject seedLexTest = new InputObject(args[8]);
        InputObject inStream = new InputObject(args[1]);
        
        for (int i = 2; i < 6; i++) {
          if (tryParse(args[i])) {
            params.add(Integer.parseInt(args[i]));
          } else {
            throw new IllegalArgumentException("Please use integers to describe parameters.");
          }
        }
        runner.run(seedLexTrain, seedLexTest, inStream, params, Integer.parseInt(args[5]), Integer.parseInt(args[6]),null,null, Integer.parseInt(args[7]));
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
  public static boolean tryParse(String s) {
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
   * @param seedLexTrain The lexicon of known words and their polarities.
   * @param sketch The sketching choice
   * @param weight The weighting choice
   */
  private void run(InputObject seedLexTrain,InputObject seedLexTest, InputObject inputStream, ArrayList<Integer> params,
                   int sketch, int weight, LearningCurve learningCurve, TaskMonitor taskMonitor, int sampleFrequency) {
    
    boolean preceiseCPUTiming = TimingUtils.enablePreciseTiming();
    long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
    
    // Read in the lexicon and give it to the trainer.
    trainer = new Trainer(evaluateStartTime,learningCurve,taskMonitor);
    trainer.initialize(seedLexTrain,seedLexTest);
    System.err.println("Vocab size: " + params.get(0) + " Context size: " + params.get(1) +
     " Window size: " + params.get(2) + " Sketching method: " + sketch + " Weighting method: " +
        weight + " Sample Frequency: "+ sampleFrequency);
    
    WordContextMatrix wcm = new WordContextMatrix(params.get(0), params.get(1),
        params.get(2), inputStream, trainer);

    // Set the sketching method
    wcm.setSketchingMethod(sketch);
    
    // Set the weighting method
    wcm.setWeightingMethod(weight);
    
    // Begin
    wcm.buildMatrix();
  }

  @Override
  protected Object doMainTask(TaskMonitor taskMonitor, ObjectRepository objectRepository) {

    ArrayList<Integer> inputParams = new ArrayList<>();


    learningCurve = new LearningCurve("classified instances");

    inputParams.add(0,vocabSizeOption.getValue());
    inputParams.add(1,contextSizeOption.getValue());
    inputParams.add(2,windowSizeOption.getValue());

    int sketchOptValue = enableHashing.isSet()?1:0;
    int weighingOptValue = enablePPMI.isSet()?1:0;

    InputObject seedLexTrain = new InputObject(SeedLexiconTrain.getValue());
    InputObject seedLexTest = new InputObject(SeedLexiconTest.getValue());
    InputObject inStream = new InputObject(InputFileName.getValue());

    run(seedLexTrain,seedLexTest,inStream,inputParams,sketchOptValue,weighingOptValue,learningCurve,taskMonitor,sampleFrequency.getValue());

    return learningCurve;
  }

  @Override
  public Class<LearningCurve> getTaskResultType() {
    return LearningCurve.class;
  }

  static void updateCurve(LearningPerformanceEvaluator evaluator, Learner model, Measurement[] measurements){
    learningCurve.insertEntry(new LearningEvaluation(measurements,evaluator,model));
//    learningCurve.
  }





}
