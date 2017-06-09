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
        ArffReader arff = new ArffReader(br);
        System.out.println("Read in data");
        final int chunk_size = 96;// 1000;//96; //172
        VOCL vocl = new VOCL(VOCL.VagueLabelMethod.CLUSTER);
        vocl.labelStream(arff.getData(), chunk_size);
    }
}
