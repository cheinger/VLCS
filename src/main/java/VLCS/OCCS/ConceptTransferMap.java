package VLCS.OCCS;

import weka.classifiers.meta.IterativeHMMClassifier;

/**
 * For concept transfer map, we build a Markov model to capture concept
 * transferpatterns during the users’ labeling process. In the following
 * subsections, we first propose a set feature extraction technique for
 * concept clustering, followed by a process of using Markov model to
 * generate a concept transfer map from the stream data
 */
public class ConceptTransferMap {
}

// probability for users to choose a particular concept given that that the prior occured
// The conditional probability transferring between two consecutive chunks Si-1 and Si
// P(Ci=cxi|Ci-1=cxi-1)
