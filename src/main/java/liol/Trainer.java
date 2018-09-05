package main.java.liol;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import moa.classifiers.Classifier;
import moa.classifiers.functions.SGD;
import moa.core.TimingUtils;

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
  private int correctlyPredicted;
  private long evaluateStartTime;
  private Object2ObjectOpenHashMap<String, String> wordPolarityMap;
  private Object2ObjectOpenHashMap<String, String> trainTestMap;
  private InstancesHeader header;
  
  public Trainer(long startTime) {
    this.evaluateStartTime = startTime;
    this.wordPolarityMap = new Object2ObjectOpenHashMap<>();
    this.trainTestMap = new Object2ObjectOpenHashMap<>();
    this.model = new SGD();
    ((SGD)model).resetLearningImpl();
    ((SGD) model).setLossFunction(1); // hinge/log/squared
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
      if (processed % 2 == 0) {
        trainTestMap.put(tokens[0], "train");
      } else {
        trainTestMap.put(tokens[0], "test");
      }
      
      processed++;
    }
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
    // model.setModelContext(header);
    // If we know the word, otherwise we ignore it and assume that we haven't seen it.
    if (wordPolarityMap.containsKey(word)) {

      if (trainTestMap.get(word).equals("train")) {
        model.trainOnInstance(inst);
      } else { // The learner is to test this one
        if (model.correctlyClassifies(inst)) {
          correctlyPredicted++;
        }
      }

      samplesSeen++;
      QueryAccuracy();
    }
  }
  
  /**
   * Prints the current accuracy of the classifier and the time that has elapsed since the start.
   */
  public void QueryAccuracy() {
    double accuracy = 100.0D * (double)this.correctlyPredicted / (double)this.samplesSeen;
    double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - this.evaluateStartTime);
    System.out.println(this.samplesSeen + " instances processed with " + accuracy + "% accuracy in " + time + " seconds.");
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
}