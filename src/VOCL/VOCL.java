package VOCL;

import weka.core.Instances;
import weka.core.Instance;

public class VOCL
{
    /**
     * This is the main VOCL method. This will apply vague one-class learning on the input stream.
     * @param stream            The stream of instances
     * @param chunk_size        The number of instances per chunk
     * @param num_classifiers   The number of classifiers forming the ensemble
     */
    public void labelStream(Instances stream, final int chunk_size, final int num_classifiers)
    {
        Chunk chunk = new Chunk(chunk_size);
        
        for (Instance inst : stream)
        {
            // Accumulate until we can form a whole chunk
            if (!chunk.addInstance(inst))
            {
                assert(chunk.size() == chunk_size);

                

                // Start new chunk (Si+1)
                chunk = new Chunk(chunk_size);
            }
        }

        // Partial chunk to process
        if (chunk.size() != 0)
        {

        }
    }
}