package moa.tasks.liol;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import moa.classifiers.Classifier;
import moa.classifiers.functions.SGD;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.TimingUtils;
import moa.core.Utils;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.tasks.TaskMonitor;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;

import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.InstanceImpl;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.Range;

/**
 * <h1>Handles the training and maintenance around the classifier and the lexicon</h1>
 *
 * @author Tristan Anderson
 * @version 1.0
 * @since 2018-08-30
 */
public class Trainer {

  protected InstancesHeader dataset;
  private int H;
  protected Random random;
  protected double W[][];

  int percentRandomProjection = 10;

  private Classifier model;
  private int samplesSeen;
  private int testSamplesSeen;
  private int correctlyPredicted;
  private long evaluateStartTime;
  private Object2ObjectOpenHashMap<String, String> wordPolarityMap;
  private Object2ObjectOpenHashMap<String, String> trainTestMap;
  private InstancesHeader header;
  private BasicClassificationPerformanceEvaluator evaluator;
  private boolean displayGraph;
  
  private int queryCounter; // To count the number of total instances seen before displaying stats.
  private int TP;
  private int TN;
  private int FP;
  private int FN;


  private DecimalFormat df;
  private LearningCurve learningCurve;
  private TaskMonitor taskMonitor;
  private double lastAcc = 0.00;

    final XYChart chart;
    final SwingWrapper<XYChart> sw;

    double[] chartY;
    int chartIndex = 0;
    Map<String,ArrayList<Double>> chartData;
    ArrayList<Double> initList;

  public Trainer(long startTime, LearningCurve learningCurve, TaskMonitor taskMonitor) {
    this.evaluateStartTime = startTime;
    this.wordPolarityMap = new Object2ObjectOpenHashMap<>();
    this.trainTestMap = new Object2ObjectOpenHashMap<>();
    this.model = new SGD();
    this.taskMonitor = taskMonitor;
    this.learningCurve = learningCurve;
    this.displayGraph = false;


    // Chart Vars
    this.chartY = new double[100];
    chartData = new HashMap<>();
    initList = new ArrayList<>();
    initList.add(0.0);

//    ((SGD)model).resetLearningImpl();
//    ((SGD)model).setLossFunction(1); // hinge/log/squared
    evaluator = new BasicClassificationPerformanceEvaluator();
    evaluator.reset();
    queryCounter = 0;

    df = new DecimalFormat("#.####");
    df.setRoundingMode(RoundingMode.CEILING);


    // Initialize Chart
    chart = new XYChartBuilder().width(800).height(600).xAxisTitle("Num of Samples").yAxisTitle("Sentiment").build();
    chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
    chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
    chart.getStyler().setYAxisLabelAlignment(Styler.TextAlignment.Left);
    chart.getStyler().setPlotMargin(0);
    chart.getStyler().setPlotContentSize(.95);
    chart.getStyler().setMarkerSize(0);
    // Display
    sw = new SwingWrapper<XYChart>(chart);
//    sw.displayChart();
  }
  
  /**
   * Sets up the known words for the system.
   * @param seedlex The seed lexicon as an input stream.
   */
  public void initialize(InputObject seedLexTrain, InputObject seedLexTest) {
    int processed = 0; // Could be a more sophisticated way of doing this.
    String line;
    String line2;

    model.prepareForUse();

    while ((line = seedLexTrain.getNextInstance()) != null) {

        String[] tokens = line.split("\t");

        if (Integer.parseInt(tokens[1]) < 0) {
          wordPolarityMap.put(tokens[0], "negative");
        } else {
          wordPolarityMap.put(tokens[0], "positive");
        }
        trainTestMap.put(tokens[0], "train");
        processed++;

      
      // TODO find a way of doing this while keeping the training and testing distributions even
      // Used python to do this...
//      if (processed < 1238) {
//      } else {
//        trainTestMap.put(tokens[0], "test");
//      }
//      if(tokens[0].equalsIgnoreCase("darkest") || tokens[0].equalsIgnoreCase("brightest")){
//          trainTestMap.put(tokens[0], "test");
//      }
      
    }

    while ((line2 = seedLexTest.getNextInstance()) != null) {

        String[] tokens = line2.split("\t");

        if (Integer.parseInt(tokens[1]) < 0) {
          wordPolarityMap.put(tokens[0], "negative");
        } else {
          wordPolarityMap.put(tokens[0], "positive");
        }

        trainTestMap.put(tokens[0], "test");

        processed++;

    }
//    ((SGD)model).prepareForUse();
  }
  
  /**
   * Sets the header of the classifier.
   * @param ih The instance header to assign
   */
  public void setHeader(InstancesHeader ih) {
    this.header = ih;


  }
  
  /**
   * Processes the incoming word and instance and updates, predicts or ignores them depending on
   * the word.
   * @param word The word to check if known
   * @param inst The instance representation of the word.
   */
  public void learn(String word, Instance inst) {
    //System.err.println(word + ": " + inst.toString());
    // If we know the word, otherwise we ignore it and assume that we haven't seen it.
    //      if(((SGD)model).getLossFunction()==0){
    //          System.out.println("Loss Function: Hinge");
    //      }
    //
    //      if(((SGD)model).getLossFunction()==1){
    //          System.out.println("Loss Function: log");
    //      }

      Instance filteredInstance = filterInstance(inst);
      if (wordPolarityMap.containsKey(word)) {
      
      // Assign the instance its class
      setInstanceClass(word, filteredInstance);

      double[] prediction = model.getVotesForInstance(filteredInstance);
      //System.err.println(Double.toString(prediction[0]) + " " + Double.toString(prediction[1]));
      if (trainTestMap.get(word).equals("train")) {
        model.trainOnInstance(filteredInstance);
      } else {
        if (Utils.maxIndex(prediction) == (int)filteredInstance.classValue()) {
          correctlyPredicted++;
          if ((int) filteredInstance.classValue() == 0) {
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
        InstanceExample eg = new InstanceExample(filteredInstance);
        evaluator.addResult(eg, prediction);

      }


//      ((SGD)model).getVotesForInstance(inst);
      queryCounter++;
      samplesSeen++;
  
      if (queryCounter == 100) {

//        System.err.println(word + " " + wordPolarityMap.get(word) + " " + trainTestMap.get(word)+
//            " " + Utils.maxIndex(prediction) + " " + prediction[Utils.maxIndex(prediction)] + "\n TP: " + TP + " FP: " + FP + "\n TN: " + TN +
//            " FN: " + FN + "\n F1: " + df.format(getF1Score()) + "\n Precision: " +
//            df.format(getPrecision()) + "\n Recall: " + df.format(getRecall()) + "\n Kappa: " + evaluator.getKappaStatistic());


          //new double[]{samplesSeen}
//          chartY[chartIndex]

//          if(false &&
//                  (trainTestMap.get(word).equals("test") && (word.equalsIgnoreCase("great"))) ||
//                          (trainTestMap.get(word).equals("test") && (word.equalsIgnoreCase("crazy")))
//                          ||
//                          (trainTestMap.get(word).equals("test") && (word.equalsIgnoreCase("nice"))) ||
//                          (trainTestMap.get(word).equals("test") && (word.equalsIgnoreCase("best"))) ||
//                          (trainTestMap.get(word).equals("test") && (word.equalsIgnoreCase("happy")))

//          ) {
//              if(!chartData.containsKey(word)){
//                  chartData.put(word,(ArrayList<Double>)initList.clone());
//                  chart.addSeries(word, chartData.get(word));
//              }
//    //              if (chart.getSeriesMap().get(word) == null) {
//    //                  chart.addSeries(word, chartY);
//    //              }
//
//              int predIndex = Utils.maxIndex(prediction);
//              double newVal = ((prediction[predIndex] * ((predIndex==1)?1:-1)) + (prediction[predIndex==1?0:1]*(((predIndex==1?0:1)==1)?1:-1)))/2.0;
////              double newVal = prediction[] * (wordPolarityMap.get(word).equals("negative") ? -1 : 1);
//    //          System.out.println("HATE + "+newVal);
//    //              chartY[chartIndex] = newVal;
//              chartData.get(word).add(newVal);
//              chart.updateXYSeries(word, null, chartData.get(word), null);
//              if((newVal < 0 && wordPolarityMap.get(word).equalsIgnoreCase("negative")) || (newVal >= 0 && wordPolarityMap.get(word).equalsIgnoreCase("positive"))){
//                  chart.getSeriesMap().get(word).setLineColor(XChartSeriesColors.BLUE);
//              }else {
//                  chart.getSeriesMap().get(word).setLineColor(XChartSeriesColors.RED);
//              }
//    //              chartIndex = (chartIndex + 1) % 100;
//
//              sw.repaintChart();
//          }



        if(this.learningCurve != null){
          ArrayList<Measurement> measurementsArray = new ArrayList<Measurement>();
          measurementsArray.add(new Measurement("Accuracy",this.lastAcc));
          measurementsArray.add(new Measurement("Precision (percent)",getPrecision()));
          measurementsArray.add(new Measurement("Kappa Statistic (percent)",evaluator.getKappaStatistic()));
          measurementsArray.add(new Measurement("Recall (percent)",getRecall()));
          measurementsArray.add(new Measurement("F1 Score (percent)",getF1Score()));
          Measurement[] result = new Measurement[measurementsArray.size()];

          MainRunner.updateCurve(evaluator,model,measurementsArray.toArray(result));
          if(taskMonitor.resultPreviewRequested()) {
            taskMonitor.setLatestResultPreview(learningCurve.copy());
          }
        }
        queryAccuracy();

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
  private void setInstanceClass(String word, Instance inst) {
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
  public void queryAccuracy() {
    //evaluator.getPerformanceMeasurements();
    //System.err.println(evaluator.getFractionCorrectlyClassified());
    double accuracy = 100.0D * (double)this.correctlyPredicted / (double)this.testSamplesSeen;

    this.lastAcc = accuracy;
    double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() -
        this.evaluateStartTime);
    System.out.println(this.testSamplesSeen+","+this.lastAcc+","+TP+","+FP+","+TN+","+FN+","+df.format(getF1Score())+","+df.format(getPrecision())+","+df.format(getRecall())+","+evaluator.getKappaStatistic()+","+time);

//    System.out.println(this.testSamplesSeen + " instances processed with " + accuracy +
//        "% accuracy in " + time + " seconds.");
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

  private void initializeRandomProjection(Instance instance) {
    this.random = new Random();

    int d = instance.numAttributes() - 1; // suppose one class attribute

    H = d * percentRandomProjection / 100;

    // initialize ReLU features
    W = new double[H][d];
    for(int j = 0; j < H; j++) {
      for(int k = 0; k < d; k++) {
        W[j][k] = this.random.nextGaussian();
      }
    }

    // initialize instance space
    Instances ds = new Instances();
    List<Attribute> v = new ArrayList<Attribute>(H);
    List<Integer> indexValues = new ArrayList<Integer>(H);

    for(int j = 0; j < H; j++) {
      v.add(new Attribute("z"+String.valueOf(j)));
      indexValues.add(j);
    }
    v.add(instance.dataset().classAttribute());
    indexValues.add(H);


    ds.setAttributes(v,indexValues);
    Range r = new Range("start-end");
    ds.setRangeOutputIndices(r);


    dataset = (new InstancesHeader(ds));
    dataset.setClassIndex(H);
    model.setModelContext(dataset);
    ((SGD)model).setLossFunction(1);

  }

  public Instance filterInstance(Instance x) {


    if(dataset==null){
      initializeRandomProjection(x);
    }

    double z_[] = new double[H+1];

    int d = x.numAttributes() - 1; // suppose one class attribute (at the end)

    for(int k = 0; k < H; k++) {
      // for each hidden unit ...
      double a_k = 0.; 								// k-th activation (dot product)
      for(int j = 0; j < d; j++) {
        a_k += (x.value(j) * W[k][j]);
      }
      z_[k] = (a_k > 0. ? a_k : 0.);				  // <------- can change threshold here
    }
    z_[H] = x.classValue();

    Instance z = new InstanceImpl(x.weight(),z_);
    z.setDataset(dataset);

    return z;
  }

}