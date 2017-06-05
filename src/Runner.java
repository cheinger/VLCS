import java.io.File;
import java.io.IOException;

import VLCS.VOCL.*;
import VLCS.OCCS.*;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Runner
{
    public static void main(String[] args) throws IOException
    {
        ArffLoader arff_loader = new ArffLoader();
        File file_data = new File(args[0]);
        arff_loader.setFile(file_data);

        Instances data = arff_loader.getDataSet();

        VOCL vocl = new VOCL();
        vocl.labelStream(data, 29000, 0);

//        for(Instance inst : data){
//            System.out.println(c.addInstance(inst));
//            System.out.println("Instance:" + inst);
//			System.out.println(inst.toString(0) + ", " + inst.toString(1));
//        }
    }
}
