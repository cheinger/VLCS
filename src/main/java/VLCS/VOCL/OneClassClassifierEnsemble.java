package VLCS.VOCL;

import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.meta.AccuracyWeightedEnsemble;
import weka.classifiers.Classifier;
import weka.classifiers.meta.OneClassClassifier;


/**
 * Created by xavier on 7/06/17.
 */
public class OneClassClassifierEnsemble extends AccuracyWeightedEnsemble {

    public void addClassifier(OneClassClassifier classifier, float weight) {
        Classifier moa_classifier = classifier.getClass().cast(Classifier.class);
        this.addToStored((moa.classifiers.Classifier) moa_classifier, weight);
    }

    public double getChunkAccuracy(Instances chunk) {
//        this.getChunkAccuracy();
//        this.
        this.currentChunk = chunk;
        this.processChunk();
        return 0.0f;
    }
}
