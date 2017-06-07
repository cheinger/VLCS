package VLCS.VOCL;

import java.util.*;
import java.util.Map.Entry;

import weka.classifiers.meta.OneClassClassifier;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class VOCL {
    public enum VagueLabelMethod {RANDOM, CLUSTER}

    private VagueLabelMethod label_method;
    private final int num_clusters = 20;
    private int positive_set_size;
    private int k = 10;
    private int class_idx;
    private int attr_idx;
    private LocalWeighting local;
    private GlobalWeighting global;
    private int[] cluster_sizes = new int[num_clusters];
    private Queue<OneClassClassifier> classifiers = new ArrayDeque<>();
    private static Remove filter = new Remove();

    public VOCL(VagueLabelMethod label_method, int positive_set_size) {
        this.label_method = label_method;
        this.positive_set_size = positive_set_size;
        this.local = new LocalWeighting(4); // Specify number of folds
        this.global = new GlobalWeighting();
        this.class_idx = 13; // FIXME
        this.attr_idx = 2; // FIXME
    }

    /**
     * This is the main VOCL method. This will apply vague one-class learning on the input stream.
     *
     * @param stream          The stream of instances
     * @param chunk_size      The number of instances per chunk
     * @param num_classifiers The number of classifiers forming the ensemble
     * @throws Exception
     */
    public void labelStream(Instances stream, final int chunk_size, final int num_classifiers) throws Exception {
        assert chunk_size >= num_clusters : "chunk size must be >= number of clusters.";

        Instances chunk = new Instances(stream);
        chunk.clear();

        for (Instance instance : stream) {
            chunk.add(instance);
            if (chunk.size() == chunk_size) {
                processChunk(chunk);
                chunk.clear();
            }
        }

        // Process partial chunk
        // processChunk(chunk);
    }

    /**
     * Main Pseudo code method from paper (Figure 5.)
     *
     * @param chunk
     * @throws Exception
     */
    private void processChunk(Instances chunk) throws Exception {

        int[] PSi = clusterVagueLabel(chunk);
        float[] WLx = local.getWeights(chunk, PSi, attr_idx, class_idx);
        float[] WGx = global.getWeights(chunk, classifiers);

        float[] Wx = calculateUnifiedWeights(WLx, WGx);
        trainNewClassifier(chunk, Wx);
    }

    private float[] calculateUnifiedWeights(float[] WLx, float[] WGx) {

        assert WLx.length == WGx.length : "Weights have different lengths.";

        // Laplace smoothing parameters
        float a = 0.5f, b = 0.5f;

        float[] Wx = new float[WLx.length];

        for (int i = 0; i < Wx.length; i++) {
            if (WLx[i] + WGx[i] > 0 ) {
                Wx[i] = (WLx[i] + WGx[i] + a) / (Math.abs(WLx[i] - WGx[i]) + b);
                assert Wx[i] >= 1 && Wx[i] <= 5 : "Unified weight expected to be between 1-5.";
            } else {
                Wx[i] = 0.f;
            }
        }

        return Wx;
    }

    /**
     * Trains a new classifier using unified weights.
     *
     * @param chunk
     * @param unified_weights
     * @throws Exception
     */
    private void trainNewClassifier(Instances chunk, float[] unified_weights) throws Exception {

        Instances new_chunk = new Instances(chunk);

        // Create new classifier to train
        OneClassClassifier new_classifier = new OneClassClassifier();
        new_classifier.setTargetClassLabel(Integer.toString(class_idx));

        // Update chunk with new weights
        for (int i = 0; i < new_chunk.size(); i++) {
            Instance instance = new_chunk.instance(i);
            instance.setWeight(unified_weights[i]);
            new_chunk.set(i, instance);
        }

        // Tell the chunk which index it's class is on
        new_chunk.setClassIndex(new_chunk.numAttributes() - 1);

        // Filter out unwanted attributes
        new_chunk = filterByAttribute(new_chunk, attr_idx);

        // Train classifier with new weighted chunk
        new_classifier.buildClassifier(new_chunk);

        // VOCL module contains the k most recent classifiers
        if (classifiers.size() == k) {
            // Full so deque last recently used
            classifiers.remove();
        }

        // Add most recently used classifier
        classifiers.add(new_classifier);
    }

    private int[] clusterVagueLabel(Instances chunk) throws Exception {
        // Cluster the data
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setPreserveInstancesOrder(true);
        kmeans.setNumClusters(num_clusters); // Highest prediction accuracy reported in paper
        kmeans.buildClusterer(chunk);
        int[] cluster_ids = kmeans.getAssignments();

        assert cluster_ids.length == chunk.size() : "KMeans didn't return an ID for each instance.";

        // Calculate cluster_sizes
        for (int i = 0; i < cluster_ids.length; ++i) {
            cluster_sizes[cluster_ids[i]]++;
        }

        // TODO replace with sorted data structure
        List<Entry<Float, Integer>> sorted_clusters = new ArrayList<Entry<Float, Integer>>();

        for (int i = 0; i < num_clusters; i++) {
            final float purity = (float) positive_set_size / cluster_sizes[i];
            sorted_clusters.add(new AbstractMap.SimpleEntry<Float, Integer>(purity, i));
        }

        // Sort clusters on purity (number of genuine positive samples in each cluster)
        Collections.sort(sorted_clusters, new Comparator<Entry<Float, Integer>>() {
                    @Override
                    public int compare(Entry<Float, Integer> o1, Entry<Float, Integer> o2) {
                        return Float.compare(o2.getKey(), o1.getKey());
                    }
                }
        );

        int[] num_pos_labels_per_clust = new int[num_clusters];
        int total_pos_labels = 0;

        for (Entry<Float, Integer> e : sorted_clusters) {
//            System.out.println(e.getKey() + " " + e.getValue() + " size: " + cluster_sizes[e.getValue()]);
            final int cluster_size = cluster_sizes[e.getValue()];
            final int num_pos = total_pos_labels + cluster_size <= positive_set_size ?
                    cluster_size :
                    positive_set_size - total_pos_labels;
            num_pos_labels_per_clust[e.getValue()] = num_pos;
            total_pos_labels += num_pos;
        }

        assert total_pos_labels == positive_set_size : "Counted positive labels incorrect.";

//        for (int i =0; i < num_clusters; i++)
//            System.out.println("pos count: " + num_pos_labels_per_clust[i]);

        // Generate positive labels
        int[] labels = new int[chunk.size()];

        for (int i = 0; i < chunk.size(); i++) {
            if (num_pos_labels_per_clust[cluster_ids[i]]-- > 0) {
                labels[i] = 1; // Mark as positive
            }
        }

//        for (int i = 0; i < chunk.size(); i++)
//        {
//            System.out.println("cluster_id: " + cluster_ids[i] + ", pos: " + labels[i]);
//        }

        // Reset for next chunk
        Arrays.fill(cluster_sizes, 0);

        return labels;
    }

    /**
     * Filters out unwanted attributes from each instance.
     *
     * @param data          The training/testing fold to filter.
     * @param attribute_idx The only attribute you want in each instance.
     * @return The filtered chunk.
     * @throws Exception
     */
    public static Instances filterByAttribute(Instances data, int attribute_idx) throws Exception {
        filter.setAttributeIndices(Integer.toString(attribute_idx)); // attr0, attr1, class (attr0 = index 2 NOT 0!)
        filter.setInputFormat(data);
        Instances new_data = Filter.useFilter(data, filter);
        new_data.setClassIndex(new_data.numAttributes() - 1);
        return new_data;
    }
}
