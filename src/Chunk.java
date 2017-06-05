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

    public boolean addInstance(Instance inst)
    {
        if (num_instances != instances.length) {
            instances[num_instances] = inst;
            num_instances++;
            return true;
        }
        return false;
    }
}
