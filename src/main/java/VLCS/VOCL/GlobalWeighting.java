package VLCS.VOCL;

import weka.classifiers.meta.OneClassClassifier;
import weka.core.Instances;

import java.util.Queue;

public class GlobalWeighting {

    public float[] getWeights(Instances chunk, Queue<OneClassClassifier> classifiers) {
        float[] weights = new float[chunk.size()];
        return weights;
    }
}
