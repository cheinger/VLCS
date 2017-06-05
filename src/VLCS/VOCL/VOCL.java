package VLCS.VOCL;

import VLCS.Chunk;

import moa.streams.ArffFileStream;
import com.yahoo.labs.samoa.instances.Instance;

public class VOCL
{
    public enum VagueLabelMethod { RANDOM, CLUSTER };

    private VagueLabelMethod label_method;
    private LocalWeighting local = new LocalWeighting();
    private GlobalWeighting global = new GlobalWeighting();
    
    public VOCL(VagueLabelMethod label_method)
    {
        this.label_method = label_method;
    }

    /**
     * This is the main VOCL method. This will apply vague one-class learning on the input stream.
     * @param stream            The stream of instances
     * @param chunk_size        The number of instances per chunk
     * @param num_classifiers   The number of classifiers forming the ensemble
     */
    public void labelStream(ArffFileStream stream, final int chunk_size, final int num_classifiers)
    {
        Chunk chunk = new Chunk(chunk_size);

        while (stream.hasMoreInstances())
        {
            Instance inst = stream.nextInstance().instance;
            // Accumulate until we can form a whole chunk
            if (!chunk.addInstance(inst))
            {
                assert(chunk.size() == chunk_size);

                processChunk(chunk);

                // Start new chunk (Si+1)
                chunk = new Chunk(chunk_size);
            }
        }

        // Partial chunk to process
        if (chunk.size() != 0)
        {
            processChunk(chunk);
        }
    }

    private void processChunk(Chunk chunk)
    {
        // Label positive instance groups
        Instance[] PSi = null;
        if (label_method == VagueLabelMethod.CLUSTER)
        {
            PSi = vagueLabelPositiveClusterGroups(chunk);
        }
        else if (label_method == VagueLabelMethod.RANDOM)
        {
            PSi = vagueLabelPositiveRandomGroups(chunk);
        }
        else
        {
            throw new IllegalArgumentException("Invalud VagueLabelType.");
        }

        // Calculate local weights for each instance in chunk
        float[] local_weights = local.weigh(chunk, 0);
    }

    private Instance[] vagueLabelPositiveClusterGroups(Chunk chunk)
    {
        return null;

    }

    private Instance[] vagueLabelPositiveRandomGroups(Chunk chunk)
    {
        return null;

    }
}
