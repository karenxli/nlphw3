# STEP 1: rename this file to textclassify_model.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
from tokenize import Double
from typing import Dict
import numpy as np
import sys
import collections
import math
from operator import itemgetter

"""
Your name and file comment here: Karen Li
"""


"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the TextClassify class here
"""
def generate_tuples_from_file(training_file_path):
  """
  Generates tuples from file formated like:
  id\ttext\tlabel
  Parameters:
    training_file_path - str path to file to read in
  Return:
    a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
  """
  f = open(training_file_path, "r", encoding="utf8")
  listOfExamples = []
  for review in f:
    if len(review.strip()) == 0:
      continue
    dataInReview = review.split("\t")
    for i in range(len(dataInReview)):
      # remove any extraneous whitespace
      dataInReview[i] = dataInReview[i].strip()
    t = tuple(dataInReview)
    listOfExamples.append(t)
  f.close()
  return listOfExamples

def precision(gold_labels, predicted_labels):
  """
  Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double precision (a number from 0 to 1)
  """
  # precision = TP/(TP + FP)

  true_pos = []
  for i in range(len(predicted_labels)):
    if(predicted_labels[i] == gold_labels[i] and int(predicted_labels[i]) == 1):
      true_pos.append(1)
  true_pos_count = len(true_pos)
  all_pos = [j for j in predicted_labels if int(j) == 1]
  all_pos_count = len(all_pos)
  if (all_pos_count == 0): return 0
  else : return float(true_pos_count/all_pos_count)

def recall(gold_labels, predicted_labels):
  """
  Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double recall (a number from 0 to 1)
  """
  # recall = TP/(TP + FN)
  intersection = []
  for i in range(len(predicted_labels)):
    if(predicted_labels[i] == gold_labels[i]):
      intersection.append(predicted_labels[i])
  true_pos = len([i for i in intersection if int(i) == 1])
  false_neg = len([j for j in gold_labels if int(j) == 1])
  if(false_neg == 0): return 0
  else : return float(true_pos / (false_neg))

def f1(gold_labels, predicted_labels):
  """
  Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double f1 (a number from 0 to 1)
  """
  # f1 = 2PR / P + R
  p = precision(gold_labels, predicted_labels)
  r = recall(gold_labels, predicted_labels)
  if(p == r == 0):
    return 0
  else: return float(2*p*r/(p + r))


"""
Implement any other non-required functions here
"""



"""
implement your TextClassify class here
"""
class TextClassify:


  def __init__(self):
    # do whatever you need to do to set up your class here
    self.unigram_labels = dict()    # every sentence with its class
    self.corpus = []                # the unigrams themselves
    self.text = []                  # examples
    self.givenLabels = []
    self.vocabulary = []            # vocabulary of the corpus (all unique words)
    self.positive = 0
    self.negative = 0
    self.posWords = dict()
    self.negWords = dict()

  # creates the bag of words
  def createBag(self):
    sentences= list(map(itemgetter(1), self.text))
    for line in sentences:
      self.corpus += line.split()
    self.givenLabels = list(map(itemgetter(2), self.text)) # collects all labels
  
  # sets P(-) and P(+)
  def overallProbability(self):
    positive_docs = len([k for k,v in self.unigram_labels.items() if float(v) == 1]) # number of docs with class 1
    negative_docs = len([k for k,v in self.unigram_labels.items() if float(v) == 0]) # number of docs with class 0
    all_docs = len(self.unigram_labels)
    
    self.positive = positive_docs/all_docs
    self.negative = negative_docs/all_docs

  # given a word, finds the probability of that word appearing depending on class
  def countProbability(self, word, classSign):
    classCount = 0
    overallCount = 0
    for i in self.unigram_labels:
      listSentence = i.split()
      if word in listSentence and int(self.unigram_labels.get(i)) == classSign:
        classCount += 1
    for j in self.unigram_labels:
      if int(self.unigram_labels.get(j)) == classSign:
        overallCount += len(j.split())
    return float((classCount + 1)/(overallCount + len(self.vocabulary)))
  
  def countClass(self):
    zero = dict()
    one = dict()

    for word in self.vocabulary:
      zero_prob = self.countProbability(word, 0)
      one_prob = self.countProbability(word, 1)
      zero[word] = zero_prob
      one[word] = one_prob
    self.posWords = one
    self.negWords = zero

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    self.text = examples
    self.createBag()
    for i in range(len(self.text)):
      self.unigram_labels[self.text[i][1]] = self.text[i][2]

    self.vocabulary = set(self.corpus)
    self.overallProbability()
    self.countClass()

  def score(self, data):
    """
    Score a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    """
    dataList = data.split()
    scores = dict()
    posScore = 1
    negScore = 1
    for word in dataList:
      if(word in self.vocabulary):
        posScore *= self.posWords.get(word)
        negScore *= self.negWords.get(word)
    scores['1'] = posScore * self.positive
    scores['0'] = negScore * self.negative

    return scores

  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    scoringResults = self.score(data)
    if(scoringResults.get('0') > scoringResults.get('1')): # what to do if equal?
      return '0' 
    else: return '1'

  def featurize(self, data):
    """
    we use this format to make implementation of your TextClassifyImproved model more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    pass

  def __str__(self):
    return "Naive Bayes - bag-of-words baseline"


class TextClassifyImproved:

  def __init__(self):
    pass

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    pass

  def score(self, data):
    """
    Score a given piece of text
    you’ll compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here
    
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    return a dictionary of the values of P(data | c)  for each class, 
    as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
    """
    pass

  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    pass

  def featurize(self, data):
    """
    we use this format to make implementation of this class more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    pass

  def __str__(self):
    return "NAME OF YOUR CLASSIFIER HERE"



def main():

  training = sys.argv[1]
  testing = sys.argv[2]

  classifier = TextClassify()
  print(classifier)
  # do the things that you need to with your base class


  # report precision, recall, f1
  

  improved = TextClassifyImproved()
  print(improved)
  # do the things that you need to with your improved class


  # report final precision, recall, f1 (for your best model)




if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
    sys.exit(1)

  main()
