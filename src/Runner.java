import java.io.File;
import java.io.IOException;
import weka.core.Instance;
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
        for(Instance inst : data){
            System.out.println("Instance:" + inst);
        }
    }
}
