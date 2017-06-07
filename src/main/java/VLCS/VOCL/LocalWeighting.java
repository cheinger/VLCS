package VLCS.VOCL;

import weka.core.Instances;

public class LocalWeighting
{
    private int num_folds;
    
    public LocalWeighting(int folds)
    {
        num_folds = folds;
    }
    
    public float[] getWeights(Instances chunk, int[] labels)
    {
        float[] weights = new float[chunk.size()];
        return weights;
    }
}
