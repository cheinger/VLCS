package VLCS.VOCL;

import weka.classifiers.meta.OneClassClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class LocalWeighting
{
    private Remove filter = new Remove();
    private int folds;

    public LocalWeighting(int folds) {
        if (folds < 3) {
            throw new IllegalArgumentException("Number of folds has to be at least 3.");
        }
        this.folds = folds;
    }
    
    public float[] getWeights(Instances chunk, int[] labels, int attribute_idx, int class_idx) throws Exception {

        assert chunk.size() % folds == 0 : "chunk size has to be multiple of num folds.";
        assert labels.length == chunk.size() : "number of labels does not match chunk size.";

        for (int i = 0; i < folds; i++)
        {
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

            // Train the one-class classifier to the specified class-idx with the training data
            OneClassClassifier occ = new OneClassClassifier();
            occ.setTargetClassLabel(Integer.toString(class_idx));
            occ.buildClassifier(new_training_set);

            for (Instance instance : new_testing_set)
            {
                double pred = occ.classifyInstance(instance);
                System.out.println("instance: " + instance + "\tpred: " + pred);
//                String className = new_training_set.attribute(new_testing_set.numAttributes() - 1).value((int)pred);
//                System.out.println("pred: " + pred + ", className: " + className);
            }

        }

        float[] weights = new float[chunk.size()];
        return weights;
    }

    /**
     * Filters out unwanted attributes from each instance.
     * @param data
     * @param attribute_idx
     * @return
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
