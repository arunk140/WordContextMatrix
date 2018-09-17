package main.java.liol;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import moa.classifiers.Classifier;
import moa.classifiers.functions.SGD;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.TimingUtils;
import moa.core.Utils;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.WindowClassificationPerformanceEvaluator;

import java.math.RoundingMode;
import java.text.DecimalFormat;

import static java.lang.Double.NaN;

/**
 * <h1>Handles the training and maintenance around the classifier and the lexicon</h1>
 *
 * @author Tristan Anderson
 * @version 1.0
 * @since 2018-08-30
 */
public class Trainer {
  
  private Classifier model;
  private int samplesSeen;
  private int testSamplesSeen;
  private int correctlyPredicted;
  private long evaluateStartTime;
  private Object2ObjectOpenHashMap<String, String> wordPolarityMap;
  private Object2ObjectOpenHashMap<String, String> trainTestMap;
  private InstancesHeader header;
  private BasicClassificationPerformanceEvaluator evaluator;
  
  private int queryCounter; // To count the number of total instances seen before displaying stats.
  private int TP;
  private int TN;
  private int FP;
  private int FN;
  
  public Trainer(long startTime) {
    this.evaluateStartTime = startTime;
    this.wordPolarityMap = new Object2ObjectOpenHashMap<>();
    this.trainTestMap = new Object2ObjectOpenHashMap<>();
    this.model = new SGD();
    ((SGD)model).resetLearningImpl();
    ((SGD) model).setLossFunction(1); // hinge/log/squared
    evaluator = new WindowClassificationPerformanceEvaluator();
    evaluator.reset();
    queryCounter = 0;
  }
  
  /**
   * Sets up the known words for the system.
   * @param seedlex The seed lexicon as an input stream.
   */
  public void Initialize(InputObject seedlex) {
    int processed = 0; // Could be a more sophisticated way of doing this.
    String line;
    
    while ((line = seedlex.getNextInstance()) != null) {
      String[] tokens = line.split("\t");
  
      if (Integer.parseInt(tokens[1]) < 0) {
        wordPolarityMap.put(tokens[0], "negative");
      } else {
        wordPolarityMap.put(tokens[0], "positive");
      }
      
      // TODO find a way of doing this while keeping the training and testing distributions even
      // Used python to do this...
      if (processed < 1238) {
        trainTestMap.put(tokens[0], "train");
      } else {
        trainTestMap.put(tokens[0], "test");
      }
      
      processed++;
    }
    
    ((SGD)model).prepareForUse();
  }
  
  /**
   * Sets the header of the classifier.
   * @param ih The instance header to assign
   */
  public void SetHeader(InstancesHeader ih) {
    this.header = ih;
    model.setModelContext(header);
  }
  
  /**
   * Processes the incoming word and instance and updates, predicts or ignores them depending on
   * the word.
   * @param word The word to check if known
   * @param inst The instance representation of the word.
   */
  public void Learn(String word, Instance inst) {
    //System.err.println(word + ": " + inst.toString());
    // If we know the word, otherwise we ignore it and assume that we haven't seen it.
    if (wordPolarityMap.containsKey(word)) {
      
      // Assign the instance its class
      SetInstanceClass(word, inst);
      
      double[] prediction = model.getVotesForInstance(inst);
      if (trainTestMap.get(word).equals("train")) {
        ((SGD)model).trainOnInstance(inst);
      } else {
        if (Utils.maxIndex(prediction) == (int)inst.classValue()) {
          correctlyPredicted++;
          if ((int) inst.classValue() == 0) {
            TN++;
          } else {
            TP++;
          }
        } else {
          if (Utils.maxIndex(prediction) == 1) {
            FP++;
          } else {
            FN++;
          }
        }
        testSamplesSeen++;
      }
      
      //evaluator.addResult(inst, prediction);

      queryCounter++;
      samplesSeen++;
  
      if (queryCounter == 10000) {
        DecimalFormat df = new DecimalFormat("#.####");
        df.setRoundingMode(RoundingMode.CEILING);
        System.err.println(word + " " + wordPolarityMap.get(word) + " " + trainTestMap.get(word)+
            " " + Utils.maxIndex(prediction) + "\n TP: " + TP + " FP: " + FP + "\n TN: " + TN +
            " FN: " + FN + "\n F1: " + df.format(getF1Score()) + "\n Precision: " +
            df.format(getPrecision()) + "\n Recall: " + df.format(getRecall()));
        QueryAccuracy();
        queryCounter = 0;
      }
    }
  }
  
  /**
   * Checks to see if the word is known and if it is, assigns the correct polarity to the word's
   * instance.
   * @param word The word to lookup
   * @param inst The instance representation of the word
   */
  private void SetInstanceClass(String word, Instance inst) {
    if (wordPolarityMap.containsKey(word)) {
      if (wordPolarityMap.get(word) == "positive") {
        inst.setClassValue(1);
      } else {
        inst.setClassValue(0);
      }
    }
  }
  
  
  
  // Additional statistics and helper methods
  
  
  
  /**
   * Prints the current accuracy of the classifier and the time that has elapsed since the start.
   */
  public void QueryAccuracy() {
    //evaluator.getPerformanceMeasurements();
    //System.err.println(evaluator.getFractionCorrectlyClassified());
    double accuracy = 100.0D * (double)this.correctlyPredicted / (double)this.testSamplesSeen;
    double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() -
        this.evaluateStartTime);
    System.out.println(this.testSamplesSeen + " instances processed with " + accuracy +
        "% accuracy in " + time + " seconds.");
  }
  
  /**
   * Calculates the current F1 score
   * @return the F1 score
   */
  private double getF1Score() {
    Double f1 = (2 * (getRecall() * getPrecision())) / (getRecall() + getPrecision());
    return (f1.isNaN()) ? 0 : f1.doubleValue();
  }
  
  /**
   * Calculates the current precision
   * @return the precision value
   */
  private double getPrecision() {
    Double precision = (double)TP / (TP + FN);
    return (precision.isNaN()) ? 0 : precision.doubleValue();
  }
  
  /**
   * Calculates the recall (AKA sensitivity)
   * @return the recall value
   */
  private double getRecall() {
    Double recall = (double)TP / (TP + FP);
    return (recall.isNaN()) ? 0 : recall.doubleValue();
  }
}