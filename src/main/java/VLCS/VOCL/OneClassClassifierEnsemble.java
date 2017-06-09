package VLCS.VOCL;

import moa.classifiers.Classifier;
import moa.classifiers.meta.WeightedMajorityAlgorithm;
import moa.core.DoubleVector;
import moa.core.InstancesHeader;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.gui.AWTRenderer;
import moa.options.ClassOption;
import moa.options.ListOption;
import moa.options.Option;
import moa.options.Options;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;
//import weka.classifiers.meta.OneClassClassifier;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

/**
 * Created by xavier on 7/06/17.
 */
//public class OneClassClassifierEnsemble extends WeightedMajorityAlgorithm {
//
//    private int curr_ensemble_size; // Used for building up ensemble in beginning
//
//    public OneClassClassifierEnsemble(int k) {
//
//        curr_ensemble_size = 0;
//
//        // Create option list of length k so that k classifiers are created
//        Option[] option_list = new Option[k];
//        for (int i = 0; i < k; i++ ) {
//            option_list[i] = new ClassOption("", ' ', "", Classifier.class, "VLCS.VOCL.MOAOneClassClassifier");
//        }
//
//        // Override default learner list options to use MOAOnceClassClassifier instead of HoeffdingTree
//        learnerListOption = new ListOption(
//                "learners",
//                'l',
//                "The learners to combine.",
//                new ClassOption("name", ' ', "", Classifier.class,
//                        "VLCS.VOCL.MOAOneClassClassifier"),
//                option_list,
//                ',');
//
//        prepareForUseImpl(new StandardTaskMonitor(), null);
//        trainingWeightSeenByModel = 1.0f;
//    }
//
//    public void addNewClassifier(MOAOneClassClassifier trained_classifier) {
//
//        if (curr_ensemble_size < ensemble.length) {
//            ensemble[curr_ensemble_size++] = trained_classifier;
//        } else {
//            throw new ArrayStoreException("Most remove last trained classifier first.");
//        }
//    }
//
//    public void removeLastRecentClassifier() {
//
//        assert curr_ensemble_size == ensemble.length : "current ensemble size must be equal to k before removing.";
//        // Shift all ensembles to left - shift out last recently trained used ensemble
//        for (int i = 0; i < ensemble.length - 1; i++) {
//            ensemble[i] = ensemble[i + 1];
//        }
//        curr_ensemble_size--;
//    }
//
//    public void updateClassifierWeights(float[] weights) {
//
//        assert curr_ensemble_size == weights.length : "Incorrect number of weights.";
//        for (int i = 0; i < weights.length; i++) {
//            ensembleWeights[i] = weights[i];
//        }
//    }
//
//    public int size() {
//        return curr_ensemble_size;
//    }
//}

public class OneClassClassifierEnsemble implements Classifier {

    private Queue<MOAOneClassClassifier> ensemble;
    private float[] ensembleWeights;

    public OneClassClassifierEnsemble(int k) {
        this.ensemble = new LinkedList<>();
        this.ensembleWeights = new float[k];
        // Initially all ensembles have equal weights
        Arrays.fill(this.ensembleWeights, 1.0f);
    }

    public void pushClassifier(MOAOneClassClassifier new_classifier) {
        if (this.ensemble.size() < this.ensembleWeights.length) {
            this.ensemble.add(new_classifier);
        } else {
            throw new ArrayStoreException("Most remove last trained classifier first.");
        }
    }

    public MOAOneClassClassifier popLastClassifier() {
        assert this.ensemble.size() == ensembleWeights.length : "current ensemble size must be equal to k before removing.";
        return this.ensemble.remove();
    }

    public Queue<MOAOneClassClassifier> getClassifiers() {
        return ensemble;
    }

    public void updateClassifierWeights(float[] weights) {
        assert this.ensemble.size() == weights.length : "Incorrect number of weights.";
        for (int i = 0; i < weights.length; i++) {
            this.ensembleWeights[i] = weights[i];
        }
    }

    public int size() {
        return this.ensemble.size();
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();
//        for(int i = 0; i < this.current_size; ++i) {
//            if(this.ensembleWeights[i] > 0.0D) {
//                try {
//                    // Use distributionForInstance instead of getVotesForInstance as we have trained
//                    // each classifier via buildClassifier()
//                    DoubleVector vote = new DoubleVector(this.ensemble[i].distributionForInstance(inst));
//                    if (vote.sumOfValues() > 0.0D) {
//                        vote.normalize();
//                        vote.scaleValues(this.ensembleWeights[i]);
//                        combinedVote.addValues(vote);
//                    }
//                } catch (Exception e) {
//                    System.err.println(e);
//                }
//            }
//        }

        return combinedVote.getArrayRef();
    }

    @Override
    public void setModelContext(InstancesHeader instancesHeader) {
        throw new UnsupportedOperationException();
    }

    @Override
    public InstancesHeader getModelContext() {
        return null;
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void setRandomSeed(int i) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean trainingHasStarted() {
        return false;
    }

    @Override
    public double trainingWeightSeenByModel() {
        return 0;
    }

    @Override
    public void resetLearning() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void trainOnInstance(Instance instance) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean correctlyClassifies(Instance instance) {
        return false;
    }

    @Override
    public Measurement[] getModelMeasurements() {
        return new Measurement[0];
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return null;
    }

    @Override
    public String getPurposeString() {
        return null;
    }

    @Override
    public Options getOptions() {
        return null;
    }

    @Override
    public void prepareForUse() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void prepareForUse(TaskMonitor taskMonitor, ObjectRepository objectRepository) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int measureByteSize() {
        return 0;
    }

    @Override
    public Classifier copy() {
        return this.copy();
    }

    @Override
    public void getDescription(StringBuilder stringBuilder, int i) {

    }

    @Override
    public String getCLICreationString(Class<?> aClass) {
        return null;
    }

    @Override
    public AWTRenderer getAWTRenderer() {
        return null;
    }
}