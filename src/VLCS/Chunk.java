package VLCS;

import weka.core.Instance;

public class Chunk
{
    private Instance[] instances;
    private int num_instances;

    public Chunk(int n)
    {
        instances = new Instance[n];
        num_instances = 0;
    }

    /**
     * Adds instances into the chunk until its full where it will then return false.
     * @param inst  The instance from the data stream
     */
    public boolean addInstance(Instance inst)
    {
        if (num_instances != instances.length) {
            instances[num_instances] = inst;
            num_instances++;
            return true;
        }
        return false;
    }

    /**
     * @return The current number of instances inside the chunk.
     */
    public int size()
    {
        return num_instances;
    }
}