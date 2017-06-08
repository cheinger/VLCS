package VLCS.VOCL;

import weka.classifiers.meta.OneClassClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Queue;

public class GlobalWeighting {

//    public float[] getWeights(Instances chunk, Queue<MOAOneClassClassifier> classifiers) throws Exception {
//
//        float[] weights = new float[chunk.size()];
//
//        for (int i = 0; i < weights.length; i++) {
//
//            float percent_predicted_positive = 0.0f;
//            Instance instance = chunk.instance(i);
//
//            System.out.println("mum classifiers: " + classifiers.size());
//
//            for (MOAOneClassClassifier classifier : classifiers) {
//                double index = classifier.classifyInstance(instance);
//                if ((int) index == Integer.parseInt(classifier.getTargetClassLabel())) {
//                    percent_predicted_positive += 1.0f;
//                }
//            }
//
//            weights[i] = percent_predicted_positive;
//
//            if (classifiers.size() > 0) {
//                weights[i] /= classifiers.size();
//            }
//            System.out.println("global weight: " + weights[i]);
//        }
//
//        return weights;
//    }
    public float[] getWeights(Instances chunk, OneClassClassifierEnsemble classifiers) throws Exception {

        float[] weights = new float[chunk.size()];

        for (int i = 0; i < weights.length; i++) {

            float percent_predicted_positive = 0.0f;
            Instance instance = chunk.instance(i);

            System.out.println("mum classifiers: " + classifiers.size());

            for (int j = 0; j < classifiers.size(); j++) {
//                if (classifiers.getSubClassifiers()[j].correctlyClassifies(instance)) {
//                    percent_predicted_positive += 1.0f;
//                }
                MOAOneClassClassifier moa_classifier = (MOAOneClassClassifier)classifiers.getSubClassifiers()[j];
                double index = moa_classifier.classifyInstance(instance);
                // If predict's as positive (part of class)
                if ((int) index == Integer.parseInt(moa_classifier.getTargetClassLabel())) {
                    percent_predicted_positive += 1.0f;
                }
            }

            weights[i] = percent_predicted_positive;

            if (classifiers.size() > 0) {
                weights[i] /= classifiers.size();
            }
            System.out.println("global weight: " + weights[i]);
        }

        return weights;
    }
}
