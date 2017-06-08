package VLCS.VOCL;

import moa.classifiers.Classifier;
import moa.classifiers.meta.WeightedMajorityAlgorithm;
import moa.options.ClassOption;
import moa.options.ListOption;
import moa.options.Option;
import moa.tasks.StandardTaskMonitor;

/**
 * Created by xavier on 7/06/17.
 */
public class OneClassClassifierEnsemble extends WeightedMajorityAlgorithm {

    private int curr_ensemble_size; // Used for building up ensemble in beginning

    public OneClassClassifierEnsemble(int k) {

        curr_ensemble_size = 0;

        // Create option list of length k so that k classifiers are created
        Option[] option_list = new Option[k];
        for (int i = 0; i < k; i++ ) {
            option_list[i] = new ClassOption("", ' ', "", Classifier.class, "VLCS.VOCL.MOAOneClassClassifier");
        }

        // Override default learner list options to use MOAOnceClassClassifier instead of HoeffdingTree
        learnerListOption = new ListOption(
                "learners",
                'l',
                "The learners to combine.",
                new ClassOption("name", ' ', "", Classifier.class,
                        "VLCS.VOCL.MOAOneClassClassifier"),
                option_list,
                ',');

        prepareForUseImpl(new StandardTaskMonitor(), null);
        trainingWeightSeenByModel = 1.0f;
    }

    public void addNewClassifier(MOAOneClassClassifier trained_classifier) {

        if (curr_ensemble_size < ensemble.length) {
            ensemble[curr_ensemble_size++] = trained_classifier;
        } else {
            throw new ArrayStoreException("Most remove last trained classifier first.");
        }
    }

    public void removeLastRecentClassifier() {

        assert curr_ensemble_size == ensemble.length : "current ensemble size must be equal to k before removing.";
        // Shift all ensembles to left - shift out last recently trained used ensemble
        for (int i = 0; i < ensemble.length - 1; i++) {
            ensemble[i] = ensemble[i + 1];
        }
        curr_ensemble_size--;
    }

    public void updateClassifierWeights(float[] weights) {

        assert curr_ensemble_size == weights.length : "Incorrect number of weights.";
        for (int i = 0; i < weights.length; i++) {
            ensembleWeights[i] = weights[i];
        }
    }

    public int size() {
        return curr_ensemble_size;
    }
}