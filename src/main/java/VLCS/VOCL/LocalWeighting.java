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
        assert chunk.size() / folds > 0 : "Must be able to fold at least once.";

        System.out.println("chunk suze: " + chunk.size());
//        for (int i = 0; i < folds; i++)
        {
            Instances training_set = new Instances(chunk);//chunk.trainCV(folds, 0);
            training_set.setClassIndex(training_set.numAttributes() - 1);
//            Instances testing_set = chunk.testCV(folds, i);
            Remove filter = new Remove();
            filter.setAttributeIndices("2"); // attr0, attr1, class (attr0 = index 2 NOT 0!)
            filter.setInputFormat(training_set);
            Instances newTrainData = Filter.useFilter(training_set, filter);
            newTrainData.setClassIndex(newTrainData.numAttributes() - 1);

            for (Instance instance : newTrainData) {
                System.out.println(instance);
            }
//
            OneClassClassifier occ = new OneClassClassifier();
//            occ.setTargetClassLabel("{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}");
            occ.setTargetClassLabel("23");
            occ.buildClassifier(newTrainData);

//            for (int j = 0; j < newTrainData.size(); j++)
//            {
//                System.out.println("newTrainData Instance: " + newTrainData.instance(j));
//            }

//            for (Instance instance : newTrainData)
            {
//                double pred = occ.classifyInstance(instance);
//                System.out.println("pred: " + pred);
            }

        }

        float[] weights = new float[chunk.size()];
        return weights;
    }
}
