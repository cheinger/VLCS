package VLCS.VOCL;

import VLCS.Chunk;

public class LocalWeighting
{
    /**
     * @param chunk     The instances to individually weigh
     * @param num_folds The number of times to fold for cross-validation
     */
    public float[] weigh(Chunk chunk, final int num_folds)
    {
        float[] weights = new float[chunk.size()];
        return weights;
    }
}
