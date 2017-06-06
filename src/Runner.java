import VLCS.VOCL.*;

import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.converters.ArffLoader.ArffReader;

public class Runner
{
    public static void main(String[] args) throws Exception
    {
        BufferedReader br = new BufferedReader(new FileReader(args[0]));
        ArffReader arff = new ArffReader(br);
        final int positive_set_size = Integer.parseInt(args[1]);
        final int chunk_size = 1000;
        VOCL vocl = new VOCL(VOCL.VagueLabelMethod.CLUSTER, positive_set_size);
        vocl.labelStream(arff.getData(), chunk_size, positive_set_size);
    }
}
