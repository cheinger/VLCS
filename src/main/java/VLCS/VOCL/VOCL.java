package VLCS.VOCL;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;

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
        this.local = new LocalWeighting(10); // Specify number of folds
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
        assert chunk_size >= num_clusters : "chunk size must be >= number of clusters.";
        
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
        int[] labels = clusterVagueLabel(chunk);
        float[] loc_weights = local.getWeights(chunk, labels);
    }

    private int[] clusterVagueLabel(Instances chunk) throws Exception
    {
        // Cluster the data
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setPreserveInstancesOrder(true);
        kmeans.setNumClusters(num_clusters); // Highest prediction accuracy reported in paper
        kmeans.buildClusterer(chunk);
        int[] cluster_ids = kmeans.getAssignments();
        
        assert cluster_ids.length == chunk.size() : "KMeans didn't return an ID for each instance.";

        // Calculate cluster_sizes
        for (int i = 0; i < cluster_ids.length; ++i)
        {
            cluster_sizes[cluster_ids[i]]++;
        }

        // TODO replace with sorted data structure
        List<Entry<Float, Integer>> sorted_clusters = new ArrayList<Entry<Float, Integer>>();
        
        for (int i = 0; i < num_clusters; i++)
        {
            final float purity = (float)positive_set_size / cluster_sizes[i];
            sorted_clusters.add(new AbstractMap.SimpleEntry<Float, Integer>(purity, i));
        }
        
        // Sort clusters on purity (number of genuine positive samples in each cluster)
        Collections.sort(sorted_clusters, (o1, o2) -> Float.compare(o2.getKey(), o1.getKey())
        );
        
        int[] num_pos_labels_per_clust = new int[num_clusters];
        int total_pos_labels = 0;
        
        for (Entry<Float, Integer> e : sorted_clusters)
        {
            System.out.println(e.getKey() + " " + e.getValue() + " size: " + cluster_sizes[e.getValue()]);
            final int cluster_size = cluster_sizes[e.getValue()];
            final int num_pos = total_pos_labels + cluster_size <= positive_set_size ?
                                cluster_size :
                                positive_set_size - total_pos_labels;
            num_pos_labels_per_clust[e.getValue()] = num_pos;
            total_pos_labels += num_pos;
        }
        
        assert total_pos_labels == positive_set_size : "Counted positive labels incorrect.";
        
        for (int i =0; i < num_clusters; i++)
            System.out.println("pos count: " + num_pos_labels_per_clust[i]);

        // Generate positive labels
        int[] labels = new int[chunk.size()];

        for (int i = 0; i < chunk.size(); i++)
        {
            if (num_pos_labels_per_clust[cluster_ids[i]]-- > 0)
            {
                labels[i] = 1; // Mark as positive
            }
        }
        
        for (int i = 0; i < chunk.size(); i++)
        {
            System.out.println("cluster_id: " + cluster_ids[i] + ", pos: " + labels[i]);
        }

        // Reset for next chunk
        Arrays.fill(cluster_sizes, 0);

        return labels;
    }
}
