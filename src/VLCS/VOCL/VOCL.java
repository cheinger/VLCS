package VLCS.VOCL;

import java.util.Random;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

public class VOCL
{
    public enum VagueLabelMethod { RANDOM, CLUSTER };

    private VagueLabelMethod label_method;
    private int positive_set_size;
    private LocalWeighting local;
    private GlobalWeighting global;

    public VOCL(VagueLabelMethod label_method, int positive_set_size)
    {
        this.label_method = label_method;
        this.positive_set_size = positive_set_size;
        this.local = new LocalWeighting();
        this.global = new GlobalWeighting();
    }

    /**
     * This is the main VOCL method. This will apply vague one-class learning on the input stream.
     * @param stream            The stream of instances
     * @param chunk_size        The number of instances per chunk
     * @param num_classifiers   The number of classifiers forming the ensemble
     * @throws Exception
     */
    public void labelStream(Instances data, final int chunk_size, final int num_classifiers) throws Exception
    {
        final int num_chunks = data.size() / chunk_size;

        // Batch process
        for (int i = 0; i < num_chunks; ++i)
        {
            Instances chunk = new Instances(data, i * chunk_size, (i + 1) * chunk_size);
            assert(chunk.size() == chunk_size);
            processChunk(chunk);
        }

        // TODO Partial chunk to process
    }

    /**
     * Main Pseudo code method from paper (Figure 5.)
     * @param chunk
     * @throws Exception
     */
    private void processChunk(Instances chunk) throws Exception
    {
        // Label positive instance groups
        Instances PSi;
        if (label_method == VagueLabelMethod.CLUSTER) {
            PSi = vagueLabelPositiveClusterGroups(chunk);
        } else if (label_method == VagueLabelMethod.RANDOM) {
            PSi = vagueLabelPositiveRandomGroups(chunk);
        } else {
            throw new IllegalArgumentException("Invalud VagueLabelType.");
        }

        // Calculate local weights for each instance in chunk
        float[] local_weights = local.weigh(chunk, 0);
    }

    private Instances vagueLabelPositiveClusterGroups(Instances chunk) throws Exception
    {
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setPreserveInstancesOrder(true);
        kmeans.setNumClusters(50); // Highest prediction accuracy reported in paper
        kmeans.buildClusterer(chunk);
        int[] cluster_ids = kmeans.getAssignments();
        
        // TODO sort clusters and data on purity (number of genuine positive samples in each cluster)
        
        return null;
    }

    private Instances vagueLabelPositiveRandomGroups(Instances chunk)
    {
        return null;
    }
}
