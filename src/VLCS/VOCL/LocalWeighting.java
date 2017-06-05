package VLCS.VOCL;

import weka.core.Instances;

public class LocalWeighting
{
    /**
     * @param chunk     The instances to individually weigh
     * @param num_folds The number of times to fold for cross-validation
     */
    public float[] weigh(Instances chunk, final int num_folds)
    {
        float[] weights = new float[chunk.size()];
        return weights;
    }
}
