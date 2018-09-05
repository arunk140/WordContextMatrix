package main.java.liol;

import cmu.arktweetnlp.Twokenize;

import com.yahoo.labs.samoa.instances.*;

import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;

import java.security.InvalidParameterException;
import java.util.Collections;
import java.util.Arrays;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.List;

import java.lang.Math;

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
  boolean isNormalized;
  boolean isPPMI;
  boolean isHashing;
  
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
        contextWordList.add(new Words(contextWordIndices.getInt(word), wr.contextDictionary.getInt(word)));
      } else {
        contextWordList.add(new Words(contextWordIndices.getInt("unk"), wr.contextDictionary.getInt(word)));
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
    
    if (isNormalized && !isPPMI) {
      attribValues = normalizer(attributeValues);
    } else if (isPPMI && !isNormalized) {
      attribValues = PMIzer(wr, attributeValues);
    } else {
      attribValues = unboxer(attributeValues);
    }
    
    return new SparseInstance(weight, attribValues, indexValues, wr.contextSize + 1);
	}
  
  /**
   * Sets the weighting method
   * @param methodNumber 0 = none, 1 = normalized, 2 = PPMI
   */
  public void setWeightingMethod(int methodNumber) {
    switch (methodNumber) {
      case 0:
        this.isNormalized = false;
        this.isPPMI = false;
        break;
      case 1:
        this.isNormalized = true;
        this.isPPMI = false;
        break;
      case 2:
        this.isNormalized = false;
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
		
		while((line = inObj.getNextInstance()) != null ) {
		  processedInstances++;
		  
			line = line.toLowerCase();
			// Tokenize the line
			List<String> tokens = Twokenize.tokenizeRawTweetText(line);

			tokensSeen += tokens.size();
      
      System.err.println(line);
			
			// Build the window
			for (int i = 0; i < tokens.size() - 1; i++) {
				int sliceStart = (i - this.windowSize >= 0) ? i - this.windowSize : 0;
				int sliceEnd = (i + this.windowSize + 1 >= tokens.size()) ? tokens.size() : i + this.windowSize + 1;
				List<String> window = tokens.subList(sliceStart, sliceEnd);
				
				WordRep focusWord = setFocusWord(tokens.get(i));
				
				// Update Context
        for (String word : window) {
          if (!contextWordIndices.containsKey(word) && contextWordIndices.size() < contextSize) {
            addToContextWordIndices(word);
          }
          if (!contextWordIndices.containsKey(word) && contextWordIndices.size() == contextSize && !word.equals(focusWord.getWord()) ||
              focusWord.equals("unk") && !contextWordIndices.containsKey(word) && contextWordIndices.size() == contextSize) {
            focusWord.addToContext("unk");
          } else if (!word.equals(focusWord.getWord())) {
            focusWord.addToContext(word);
          }
        }
        
				focusWord.incrementTweets();
        //masterCtxChecker(); // Check that the context map is correct
				Instance sprseFocus = SparseCreator(focusWord);
        InstancesHeader instHeader = createInstanceHeader();
        sprseFocus.setDataset(instHeader);
        
        //System.err.println(focusWord.getWord() + " " + sprseFocus.toString());
        
        // If the word has been seen a significant (10) number of times, send it to be classified.
        if (focusWord.numTweets >= 10) {
        	trainer.SetHeader(instHeader);
          trainer.Learn(focusWord.getWord(), sprseFocus);
        }
			}
			System.err.println();
		}
		System.err.println("Program ran to completion");
	}
	
	private WordRep setFocusWord(String word) {
		WordRep fWord;
		// Word already seen
		if (this.vocabulary.containsKey(word)) {
			fWord = this.vocabulary.get(word);
		} else if (this.vocabulary.size() != vocabSize) {
			fWord = new WordRep(word, this.contextSize);
			addToVocab(fWord);
		} else {
			fWord = vocabulary.get("unk");
		}
		return fWord;
	}
	
	private void addToVocab(WordRep wr) {
		if (this.vocabulary.size() != this.vocabSize && !this.vocabulary.containsKey(wr.word)) {
			if (this.vocabulary.size() + 1 == this.vocabSize) {
				wr.setWord("unk");
				this.vocabulary.put("unk", wr);
			} else {
				this.vocabulary.put(wr.word, wr);
			}
		}
	}
	
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
      attributes.add(new Attribute("context" + (i + 1)));
    }
  
    attributes.add(new Attribute("class", classLabels));
    Instances insts = new Instances("word-context", attributes, 0);
    InstancesHeader header = new InstancesHeader(insts);
    header.setClassIndex(header.numAttributes() - 1);
    return header;
  }
  
  private void masterCtxChecker() {
    for (String key : contextWordIndices.keySet()) {
      System.err.println(key + " : " + contextWordIndices.getInt(key));
    }
  }
  
  /**
   * Unboxes a Double array to its primitive form.
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
   * Converts a Double array into a normalized double array
   * @param attribs The Double array of attribute values
   * @return A normalized double array
   */
  private double[] normalizer(Double[] attribs) {
      double[] normalizedAttribs = new double[attribs.length];
      double max = Collections.max(Arrays.asList(attribs));
      
      for (int i = 0; i < attribs.length; i++) {
				normalizedAttribs[i] = attribs[i] / max;
      }
      
    return normalizedAttribs;
	}
  
  /**
   * Converts a Double array into a normalized double array but ignores the presence of unknown
   * words.
   * @param attribs The Double array of attribute values
   * @param indexes The index
   * @return A normalized double array
   */
	private double[] normalizerIgnUnk(Double[] attribs, int[] indexes) {
    double[] normalizedAttribs = new double[attribs.length];
    // If the unknown word is in the attribute array let it have no significance
    for (int i : indexes) {
      if (i == 0) {
        attribs[0] = (double) 0;
      }
    }
    double max = Collections.max(Arrays.asList(attribs));
    for (int i = 0; i < attribs.length; i++) {
      normalizedAttribs[i] = attribs[i] / max;
    }
    
    return normalizedAttribs;
  }
	
	/**
	 * Converts a Double array into a double array of PMI values
	 * @param attribs The Double array of attribute values
	 * @return A PMI'd double array
	 */
	private double[] PMIzer(WordRep wr, Double[] attribs) {
		double[] PMIAttribs = new double[attribs.length];
    for (int i = 0; i < attribs.length; i++) {
      int contextWordCount = vocabulary.get(attribs[i]).numTweets;
      double pmiRes = (attribs[i] * tokensSeen) / (wr.numTweets * contextWordCount);
      // Log base 2
      double res = Math.log(pmiRes) / Math.log(2);
      PMIAttribs[i] = max(0.0, res);
    }

    return PMIAttribs;
  }
 
	private class WordRep {
		String word;
		double polarity;
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
		
		public double getPolarity() {
			return polarity;
		}
		
		public void setPolarity(double p) {
			polarity = p;
		}
		
		public void addToContext(String contextWord) {
			if (contextDictionary.containsKey(contextWord) || isFull) {
				if (contextDictionary.containsKey(contextWord)) {
					contextDictionary.put(contextWord, contextDictionary.getInt(contextWord) + 1);
				} else {
					contextDictionary.put("unk", contextDictionary.getInt("unk") + 1);
				}
			} else if (contextDictionary.size()  + 1 == contextSize) {
				contextDictionary.put("unk", 1);
				isFull = true;
			} else {
				contextDictionary.put(contextWord, 1);
			}
		} 
	}
	
	private class Words implements Comparator<Words> {
    private Integer idx;
    private double value;
    
    public Words() {}
    
    public Words(Integer i, double v) {
      idx = i;
      value = v;
    }
    
    public Integer getIdx() { return idx; }
    public double getValue() { return value; }
    
    public int compare(Words origWord, Words otherWord) {
      return origWord.getIdx().compareTo(otherWord.getIdx());
    }
  }
}