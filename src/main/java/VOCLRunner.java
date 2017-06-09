import VLCS.VOCL.*;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.meta.OneClassClassifier;
import weka.core.converters.ArffLoader.ArffReader;

public class VOCLRunner
{
    public static void main(String[] args) throws Exception
    {
        BufferedReader br = new BufferedReader(new FileReader(args[0]));
        int class_index = Integer.parseInt(args[1]);
        int attr_index = Integer.parseInt(args[2]);
        ArffReader arff = new ArffReader(br);
        final int chunk_size = 1000;
        VOCL vocl = new VOCL(VOCL.VagueLabelMethod.CLUSTER, class_index, attr_index);
        vocl.labelStream(arff.getData(), chunk_size);
    }
}
