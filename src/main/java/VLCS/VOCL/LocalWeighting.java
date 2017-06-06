package VLCS.VOCL;

import weka.classifiers.meta.OneClassClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.beans.CrossValidationFoldMaker;

public class LocalWeighting
{
    private int folds;

    public LocalWeighting(int folds) {
        if (folds < 3) {
            throw new IllegalArgumentException("Number of folds has to be at least 3.");
        }
        this.folds = folds;
    }
    
    public float[] getWeights(Instances chunk, int[] labels) throws Exception {

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

            Remove filter = new Remove();
            filter.setAttributeIndices("2"); // attr0, attr1, class (attr0 = index 2 NOT 0!)
            filter.setInputFormat(training_set);
            Instances new_training_set = Filter.useFilter(training_set, filter);
            new_training_set.setClassIndex(new_training_set.numAttributes() - 1);

            OneClassClassifier occ = new OneClassClassifier();
            occ.setTargetClassLabel("13");
            occ.buildClassifier(new_training_set);

            Remove filter2 = new Remove();
            filter2.setAttributeIndices("2"); // attr0, attr1, class (attr0 = index 2 NOT 0!)
            filter2.setInputFormat(testing_set);
            Instances new_testing_set = Filter.useFilter(testing_set, filter2);
            new_testing_set.setClassIndex(new_testing_set.numAttributes() - 1);

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
}
