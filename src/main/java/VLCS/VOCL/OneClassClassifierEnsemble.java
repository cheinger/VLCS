package VLCS.VOCL;

import moa.classifiers.meta.AccuracyWeightedEnsemble;
import moa.options.ClassOption;
import moa.options.FloatOption;
import weka.classifiers.Classifier;

/**
 * Created by xavier on 7/06/17.
 */
public class OneClassClassifierEnsemble extends AccuracyWeightedEnsemble {

    public OneClassClassifierEnsemble() {
        learnerOption = new ClassOption("learner", 'l', "Classifier to train.", Classifier.class, "VLCS.VOCL.MOAOneClassClassifier");
        memberCountOption = new FloatOption("memberCount", 'n', "The maximum number of classifier in an ensemble.", 10, 1, Integer.MAX_VALUE);
        storedCountOption = new FloatOption("storedCount", 'r', "The maximum number of classifiers to store and choose from when creating an ensemble.", 10.0D, 1.0D, 2.147483647E9D);
        this.prepareForUse();
    }

    public void addClassifier(MOAOneClassClassifier classifier, float weight) {
        this.addToStored(classifier, weight);

        System.out.println("STORE LENGTH: " + this.storedLearners.length + ", MAX_STORED_COUNT: " + maxStoredCount);
    }

//    public double majorityVoteClassifierInstance(Instance instance) {
//        for (int i = 0; i < this.storedLearners.length; i++) {
//            System.out.println("FDSFDSFS");
//        }
//        return 0.f;
//    }
}
