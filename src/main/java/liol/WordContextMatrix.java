package liol;

import cmu.arktweetnlp.Twokenize;
import com.yahoo.labs.samoa.instances.*;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;

import javax.lang.model.element.UnknownElementException;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static java.lang.Double.max;

/**
 * <h1>Handles a word-context matrix</h1>
 *
 * Maintains the development and construction of word context vectors.
 * Also maintains the classes that it uses (WordRep and Words)
 *
 * @author Tristan Anderson
 * @version 1.0
 * @since 2018-08-30
 */
public class WordContextMatrix {

  protected Object2ObjectMap<String, WordRep> vocabulary;
  protected int processedInstances;

  private int tokensSeen;
  private Object2IntMap<String> contextWordIndices;
  private int nextPos;
  private int vocabSize;
  private int contextSize;
  private int windowSize;
  private InputObject inObj;
  private Trainer trainer;
  boolean isPPMI;
  boolean isHashing;

  private Object2IntMap<String> contextBinCounts; // To keep track of the overall bin counts...

  /**
   * The constructor. Initializing the WCM.
   * @param vSize The vocabulary size
   * @param cSize The context vector size
   * @param wSize The window size
   * @param inStream The input stream
   */
  public WordContextMatrix(int vSize, int cSize, int wSize, InputObject inStream, Trainer trainer) {
    this.windowSize = wSize;
    this.vocabSize = vSize;
    this.contextSize = cSize;
    // The input stream
    this.inObj = inStream;
    this.vocabulary = new Object2ObjectOpenHashMap<>();
    this.contextWordIndices = new Object2IntOpenHashMap<>();
    this.nextPos = 1;
    this.trainer = trainer;

    // Set the weighting style (default none)
    setWeightingMethod(0);
    // Set the sketching style (default none)
    setSketchingMethod(0);

    addToVocab("unk");
    //vocabulary.get("unk").numTweets = 0; // Special case since it's just to put the word there.
  }

  /**
   * Creates a MOA sparse instance from the representation of the word.
   *
   * Inner details (that were missing from the original MOA documentation) for the sparse instance:
   * weight: the instance's weight
   * attributeValues: the vector of attribute values (only the ones to be stored)
   * indexValues: The indices of the values as they would appear in a full vector
   * contextSize/maxValues: the maximum number of values that can be stored
   *
   * @param wr The word representations
   * @return The sparse instance
   */
  private SparseInstance SparseCreator(WordRep wr) {
    double weight = 1;
    Double[] attributeValues = new Double[wr.contextDictionary.size() + 1];
    int[] indexValues = new int[wr.contextDictionary.size() + 1];

    List<Words> contextWordList = new ArrayList<>();

    for (String word: wr.contextDictionary.keySet()) {
      // If the word isn't unk
      if (contextWordIndices.containsKey(word)) {
        contextWordList.add(new Words(contextWordIndices.getInt(word),
            wr.contextDictionary.getInt(word), word));
      } else {
        if (isHashing) {
          throw new UnknownElementException(null, word);
        } else {
          contextWordList.add(new Words(contextWordIndices.getInt("unk"),
              wr.contextDictionary.getInt(word), "unk"));
        }
      }
    }

    Words[] sortedAttribs = new Words[contextWordList.size()];
    sortedAttribs = contextWordList.toArray(sortedAttribs);
    Arrays.sort(sortedAttribs, new Words());

    // Add to the two arrays
    for (int i = 0; i < sortedAttribs.length; i++) {
      attributeValues[i] = sortedAttribs[i].value;
      indexValues[i] = sortedAttribs[i].idx;
    }

    // Set the class position to be the last one
    indexValues[indexValues.length - 1] = contextSize; // Remember that it's 0 indexed!
    attributeValues[indexValues.length - 1] = Double.NaN;

    // Remove the assignment when the if statements are complete
    double[] attribValues;

    if (isPPMI) {
      attribValues = PPMIzer(wr, sortedAttribs);
      attribValues[indexValues.length - 1] = Double.NaN;
    } else {
      attribValues = unboxer(attributeValues);
    }

    return new SparseInstance(weight, attribValues, indexValues, wr.contextSize + 1);
  }

  /**
   * Sets the weighting method
   * @param methodNumber 0 = none, 1 = PPMI
   */
  public void setWeightingMethod(int methodNumber) {
    switch (methodNumber) {
      case 0:
        this.isPPMI = false;
        break;
      case 1:
        this.isPPMI = true;
        break;
      default:
        throw new InvalidParameterException();
    }
  }

  /**
   * Sets the sketching method
   * @param methodNumber 0 = none, 1 = hashing
   */
  public void setSketchingMethod(int methodNumber) {
    switch (methodNumber) {
      case 0:
        this.isHashing = false;
        break;
      case 1:
        this.isHashing = true;
        prepareForHashing();
        break;
      default:
        throw new InvalidParameterException();
    }
  }

  /**
   * Builds the matrix of sparse vectors by incrementally updating the word vectors.
   * Does this by tokenizing and pre-processing all the tweets/sentences and then sliding a window
   * across it.
   * Ends by producing a sparse instance and outputting it to an arff file.
   */
  public void buildMatrix() {
    String line;

    System.err.println("Program started...");

    while((line = inObj.getNextInstance()) != null ) {
      processedInstances++;

      line = line.toLowerCase();
      // Tokenize the line
      List<String> tokens = Twokenize.tokenizeRawTweetText(line);

      tokensSeen += tokens.size(); // For PPMI among other things

      // Add to vocab
      for (String word: tokens) {
        if (isHashing) {
          int binId = Math.abs(jenkinsHash(word.getBytes()) % contextSize);
          contextBinCounts.put("contextbin" + binId, contextBinCounts.getInt("contextbin" + binId) + 1);
          addToVocab(word);
        } else {
          addToVocab(word);
        }
      }

      //System.err.println(line);

      // Build the window
      for (int i = 0; i < tokens.size() - 1; i++) {
        int sliceStart = (i - this.windowSize >= 0) ? i - this.windowSize : 0;
        int sliceEnd = (i + this.windowSize + 1 >= tokens.size()) ?
            tokens.size() : i + this.windowSize + 1;
        List<String> window = tokens.subList(sliceStart, sliceEnd);

        WordRep focusWord = getWordRep(tokens.get(i));

        // Update Context
        if (isHashing) {
          for (String word : window) {
            if (!word.equals(focusWord.getWord())) {
              int binId = Math.abs(jenkinsHash(word.getBytes()) % contextSize);
              // Increment the binId in the context map.
              focusWord.addToContext("contextbin" + binId);
            }
          }
        } else {
          for (String word : window) {
            if (!contextWordIndices.containsKey(word) && contextWordIndices.size() < contextSize) {
              addToContextWordIndices(word);
            }
            if (!contextWordIndices.containsKey(word) && contextWordIndices.size() == contextSize &&
                !word.equals(focusWord.getWord()) || focusWord.equals("unk") &&
                !contextWordIndices.containsKey(word) && contextWordIndices.size() == contextSize) {
              focusWord.addToContext("unk");
            } else if (!word.equals(focusWord.getWord())) {
              focusWord.addToContext(word);
            }
          }
        }

        //focusWord.incrementTweets();

        //masterCtxChecker(); // Check that the context map is correct

        // If the word has been seen a significant (10) number of times, send it to be classified.
        if (focusWord.numTweets >= 1) {
          Instance sprseFocus = SparseCreator(focusWord);
          InstancesHeader instHeader = createInstanceHeader();
          sprseFocus.setDataset(instHeader);
          //System.err.println(focusWord.getWord() + " " + sprseFocus.toString());

          trainer.SetHeader(instHeader);
          trainer.Learn(focusWord.getWord(), sprseFocus);
        }
      }
      //System.err.println();
    }
    System.err.println("Program ran to completion");
  }

  /**
   * Returns the word rep for the given word, or if it doesn't exist in the vocab, returns unk
   * @param word The word to lookup
   * @return The wordrep for the given word or for unk
   */
  private WordRep getWordRep(String word) {
    return (this.vocabulary.containsKey(word)) ? this.vocabulary.get(word) : vocabulary.get("unk");
  }

  /**
   * Adds a word representation to the vocabulary if it doesn't exist already and the vocab isn't full
   * @param word The word to add to the vocabulary
   */
  private void addToVocab(String word) {
    if (this.vocabulary.size() != this.vocabSize && !this.vocabulary.containsKey(word)) {
      this.vocabulary.put(word, new WordRep(word, this.contextSize));
      this.vocabulary.get(word).incrementTweets();
    }
  }

  /**
   * Checks to see if we have space for a new word and if so add it, otherwise add "unk"
   * @param contextWord The word to add to the context index map
   */
  private void addToContextWordIndices(String contextWord) {
    if (contextWordIndices.size() == 0) {
      contextWordIndices.put("unk", 0);
    }
    if (this.contextWordIndices.size() != this.contextSize &&
        !this.contextWordIndices.containsKey(contextWord)) {
      this.contextWordIndices.put(contextWord, nextPos);
      nextPos++;
    }
  }

  /**
   * For use with the hashing implementation, we don't need to worry about the words, just the bins.
   * Populates the context word indices map with the names of the bins.
   */
  private void prepareForHashing() {
    this.contextBinCounts = new Object2IntOpenHashMap<>();
    // Initialize the context word indices for hashing.
    for (int i = 0; i < contextSize; i++) {
      contextWordIndices.put("contextbin" + i, i);
      contextBinCounts.put("contextbin" + i, 0);
    }
  }

  /**
   * Creates a MOA instances header with two class options and a size that
   * holds contextSize attributes.
   *
   * @return a blank / default instances header
   */
  private InstancesHeader createInstanceHeader() {
    ArrayList<Attribute> attributes = new ArrayList<>();
    ArrayList<String> classLabels = new ArrayList<>();

    classLabels.add("negative");
    classLabels.add("positive");

    for (int i = 0; i < contextSize; i++) {
      if(isHashing) {
        attributes.add(new Attribute("contextbin" + (i + 1)));
      } else {
        attributes.add(new Attribute("context" + (i + 1)));
      }
    }

    attributes.add(new Attribute("class", classLabels));
    Instances insts = new Instances("word-context", attributes, 0);
    InstancesHeader header = new InstancesHeader(insts);
    header.setClassIndex(header.numAttributes() - 1);
    return header;
  }

  /**
   * Checks the master context dictionary's content.
   */
  private void masterCtxChecker() {
    for (String key : contextWordIndices.keySet()) {
      System.err.println(key + " : " + contextWordIndices.getInt(key));
    }
  }

  /**
   * Unboxes a Double array to its primitive form.
   * Done because as far as I'm aware, there is no way to do this for an array of Doubles in Java.
   * @param attribs The Double array (array of boxed doubles)
   * @return An unboxed primitive double array
   */
  private double[] unboxer(Double[] attribs) {
    double[] d = new double[attribs.length];
    for (int i = 0; i < attribs.length; i++) {
      d[i] = attribs[i];
    }
    return d;
  }

  /**
   * Converts a Double array into a double array of PMI values
   * @param attribs The Double array of attribute values
   * @return A PMI'd double array
   */
  private double[] PPMIzer(WordRep wr, Words[] attribs) {
    double[] PMIAttribs = new double[attribs.length + 1];
    for (int i = 0; i < attribs.length; i++) {
      if (!vocabulary.containsKey(attribs[i].word) && !isHashing) {
        System.err.println(attribs[i].word + ": " + wr.getWord());
      }
      int contextWordCount = (isHashing) ?
          contextBinCounts.getInt(attribs[i].word) : vocabulary.get(attribs[i].word).numTweets;
      double pmiRes = (attribs[i].value * tokensSeen) / (wr.numTweets * contextWordCount);
      // Log base 2
      double res = Math.log(pmiRes) / Math.log(2);
      PMIAttribs[i] = max(0.0, res);
    }

    return PMIAttribs;
  }

  /**
   * Hashes a string (context word) and returns its hash.
   * @param key the byte array of the word
   * @return Int result of hashing the byte array rep of the word
   */
  private int jenkinsHash(byte[] key) {
    int i = 0;
    int hash = 0;
    while (i != key.length) {
      hash += key[i++];
      hash += hash << 10;
      hash ^= hash >> 6;
    }
    hash += hash << 3;
    hash ^= hash >> 11;
    hash += hash << 15;
    return hash;
  }

  private class WordRep {
    String word;
    int contextSize;
    Object2IntMap<String> contextDictionary;
    Boolean isFull = false;
    int numTweets = 0;

    public WordRep(String word, Integer maxContextSize) {
      setWord(word);
      setContextSize(maxContextSize);
      this.contextDictionary = new Object2IntOpenHashMap<>();
    }

    public String getWord() {
      return word;
    }

    private void setWord(String w) {
      word = w;
    }

    public void setContextSize(int c) {
      contextSize = c;
    }

    public void incrementTweets() {
      numTweets++;
    }

    public void addToContext(String contextWord) {
      if (contextDictionary.containsKey(contextWord) || isFull) {
        if (contextDictionary.containsKey(contextWord)) {
          contextDictionary.put(contextWord, contextDictionary.getInt(contextWord) + 1);
        } else {
          if (isHashing) {
            throw new RuntimeException("Assigned context word is out of range");
          } else {
            contextDictionary.put("unk", contextDictionary.getInt("unk") + 1);
          }
        }
      } else if (contextDictionary.size() + 1 == contextSize) {
        if (isHashing) {
          contextDictionary.put(contextWord, 1);
        } else {
          contextDictionary.put("unk", 1);
        }
        isFull = true;
      } else {
        contextDictionary.put(contextWord, 1);
      }
    }
  }

  private class Words implements Comparator<Words> {
    private Integer idx;
    private double value;
    private String word;

    public Words() {}

    public Words(Integer i, double v, String w) {
      idx = i;
      value = v;
      word = w;
    }

    public Integer getIdx() { return idx; }
    public double getValue() { return value; }
    public String getWord() { return word; }

    public int compare(Words origWord, Words otherWord) {
      return origWord.getIdx().compareTo(otherWord.getIdx());
    }
  }
}