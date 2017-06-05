import VLCS.VOCL.*;
import VLCS.OCCS.*;

import java.io.IOException;
import moa.streams.ArffFileStream;

public class Runner
{
    public static void main(String[] args) throws IOException
    {
        ArffFileStream arff_stream = new ArffFileStream(args[0], -1);

        VOCL vocl = new VOCL();
        vocl.labelStream(arff_stream, 29000, 0);

        // while (arff_stream.hasMoreInstances())
        // {
        //     InstanceExample inst = arff_stream.nextInstance();
        //     System.out.println("Instance:" + inst.instance);
        // }

    //    for(Instance inst : data){
    //        System.out.println(c.addInstance(inst));
    //        System.out.println("Instance:" + inst);
	// 		System.out.println(inst.toString(0) + ", " + inst.toString(1));
    //    }
    }
}
