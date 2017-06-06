package VLCS.VOCL;

import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;

public class VOCL
{
    public enum VagueLabelMethod { RANDOM, CLUSTER };

    private VagueLabelMethod label_method;
    private final int num_clusters = 20;
    private int[] cluster_sizes = new int[num_clusters];
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
    public void labelStream(Instances stream, final int chunk_size, final int num_classifiers) throws Exception
    {
         Instances chunk = new Instances(stream);
         chunk.clear();
         
         for (Instance instance : stream)
         {
             chunk.add(instance);
             if (chunk.size() == chunk_size)
             {
                 processChunk(chunk);
                 chunk.clear();
             }
         }
         
         // Process partial chunk
         // processChunk(chunk);
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
        kmeans.setNumClusters(num_clusters); // Highest prediction accuracy reported in paper
        kmeans.buildClusterer(chunk);
        int[] cluster_ids = kmeans.getAssignments();
        
        TreeMap<Float, Integer> cluster_purity_id = new TreeMap<Float, Integer>(Collections.reverseOrder());

        // Calculate cluster_sizes
        for (int i = 0; i < cluster_ids.length; ++i) {
            cluster_sizes[cluster_ids[i]]++;
        }
        
        // Sort clusters on purity (number of genuine positive samples in each cluster)
        for (int i = 0; i < num_clusters; ++i) {
            cluster_purity_id.put((float)positive_set_size / cluster_sizes[cluster_ids[i]], cluster_ids[i]);
        }
        
        int[] cluster_pos_num_labels = new int[num_clusters];
        int total_pos_labels = 0;
        int itr = 0;
        for (Map.Entry<Float, Integer> e : cluster_purity_id.entrySet()) {
            final int cluster_size = cluster_sizes[e.getValue()];
            final int num_pos = total_pos_labels + cluster_size <= positive_set_size ?
                                cluster_size : 
                                positive_set_size - total_pos_labels;
            cluster_pos_num_labels[itr++] = num_pos;
            total_pos_labels += num_pos;
        }

        for (int i = 0; i <num_clusters; i++ ) {
            System.out.println(i + " " + cluster_pos_num_labels[i]);
        }

        
        
        // Reset cluster_sizes for next chunk
        for (int i = 0; i < num_clusters; i++) {
            cluster_sizes[i] = 0;
        }
        
        return null;
    }

    private Instances vagueLabelPositiveRandomGroups(Instances chunk)
    {
        return null;
    }
}
