package VLCS.VOCL;

import weka.classifiers.meta.OneClassClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class LocalWeighting {
    private Remove filter = new Remove();
    private int folds;

    public LocalWeighting(int folds) {
        if (folds < 3) {
            throw new IllegalArgumentException("Number of folds has to be at least 3.");
        }
        this.folds = folds;
    }

    /**
     * Local Weighting Pseudo code method from paper (Figure 6.)
     *
     * @param chunk
     * @param labels
     * @param attribute_idx
     * @param class_idx
     * @return
     * @throws Exception
     */
    public float[] getWeights(Instances chunk, int[] labels, int attribute_idx, int class_idx) throws Exception {

        assert chunk.size() % folds == 0 : "chunk size has to be multiple of num folds.";
        assert labels.length == chunk.size() : "number of labels does not match chunk size.";

        // TODO Not sure whether we need a 'chunk_classifier'
        OneClassClassifier chunk_classifier = new OneClassClassifier();
        chunk_classifier.setTargetClassLabel(Integer.toString(class_idx));

        OneClassClassifier[] fold_classifiers = new OneClassClassifier[folds];

        float[] weights = new float[chunk.size()];

        for (int i = 0; i < folds; i++) {
            fold_classifiers[i] = new OneClassClassifier();
            // Specify attribute interest
            fold_classifiers[i].setTargetClassLabel(Integer.toString(class_idx));

            Instances training_set = chunk.trainCV(folds, i);
            Instances testing_set = chunk.testCV(folds, i);

            // Tell the sets which index indicates the class
            training_set.setClassIndex(training_set.numAttributes() - 1);
            testing_set.setClassIndex(testing_set.numAttributes() - 1);

            System.out.println("folds: " + folds + ", chunk_size: " + training_set.size());

            Instances new_training_set = filterByAttribute(training_set, attribute_idx);
            Instances new_testing_set = filterByAttribute(testing_set, attribute_idx);

            assert new_training_set.size() == new_training_set.size() : "filtering changed the size of the set.";
            assert new_testing_set.size() == testing_set.size() : "filtering changed the size of the set.";

            // Train the one-class classifiers to the specified class-idx with the training data
            fold_classifiers[i].buildClassifier(new_training_set);
            chunk_classifier.buildClassifier(new_training_set);

            for (int p = 0; p < new_testing_set.size(); p++) {
                Instance instance = new_testing_set.instance(p);
                double index = chunk_classifier.classifyInstance(instance);
                // Update for positive samples
                if (labels[p] == 1) {
                    // If Classifies p as positive then set weight to 1.0 else 0.5;
                    weights[p] = ((int) index == class_idx) ? 1.0f : 0.5f;
                    System.out.println("instance: " + instance + "\tindex: " + index);
                    // Update for unlabelled samples that are Classified as positive
                } else if (labels[p] == 0 && (int) index == class_idx) {
                    // Predict unlabelled instance using all <= i OneClassClassifiers to calculate weight
                    float weight = 0.f;
                    for (int f = 0; f < i; f++) {
                        double sub_index = fold_classifiers[f].classifyInstance(instance);
                        // Increase weight if positive
                        weight += ((int) sub_index == class_idx) ? 1.0f : 0.0f;
                    }
                    weights[p] = (weight / folds) + 1;
//                    System.out.println("unlabelled -> instance: " + instance + "\taverge index: " + weights[p]);
                }
            }

        }

        // Normalize weight values to 0-1 for unlabelled set (USi)
        for (int p = 0; p < weights.length; p++) {
            if (labels[p] == 0) {
//                float normalized = (x-min(x))/(max(x)-min(x));
//              weights[p] = (x-min(x))/(max(x)-min(x));
//                System.out.println("normalize: " + weights[p]);
            }
        }

        return weights;
    }

    /**
     * Filters out unwanted attributes from each instance.
     *
     * @param data          The training/testing fold to filter.
     * @param attribute_idx The only attribute you want in each instance.
     * @return The filtered chunk.
     * @throws Exception
     */
    private Instances filterByAttribute(Instances data, int attribute_idx) throws Exception {
        filter.setAttributeIndices(Integer.toString(attribute_idx)); // attr0, attr1, class (attr0 = index 2 NOT 0!)
        filter.setInputFormat(data);
        Instances new_data = Filter.useFilter(data, filter);
        new_data.setClassIndex(new_data.numAttributes() - 1);
        return new_data;
    }
}
