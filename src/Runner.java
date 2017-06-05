import VLCS.VOCL.*;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.converters.ArffLoader.ArffReader;


public class Runner
{
    public static void main(String[] args) throws Exception
    {
        BufferedReader br = new BufferedReader(new FileReader(args[0]));
        ArffReader arff = new ArffReader(br);
        VOCL vocl = new VOCL(VOCL.VagueLabelMethod.CLUSTER);
        vocl.labelStream(arff.getData(), 29000, 0);
    }
}
