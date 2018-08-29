# Learning an Incremental Opinion Lexicon from Twitter Streams

This project takes as input (and strong assumption) a seed lexicon of known words with their
polarities and a file of arbitrary size to simulate an input stream. The program then builds
a word context matrix from the input stream by sliding a window over the input. Word vectors
are formed from this process. When a word has been seen a specified number of times, it is
sent to a classifier to be trained or tested. 

It is hoped that this application and implementation of common techniques in a novel manner 
will yield comparable results to established methods but will use significantly less resources
to do so.

## Required files and inputs

* Seed lexicon (words and polarities)
* File for streaming of tweets (the content)

## Usage

When compiled, the program is run from the command line by passing in the following arguemnts:

[SeedLexicon][InputFile][OutputFileName][VocabularySize][ContextVectorSize][WindowSize]
