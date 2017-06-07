package VLCS.VOCL;

import java.util.*;
import java.util.Map.Entry;

import meka.classifiers.multitarget.meta.EnsembleMT;
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
    private int k = 10;
    private int class_idx;
    private int attr_idx;

    private VagueLabeling vague_clustering = new ClusterVagueLabeling();

    private LocalWeighting local;
    private GlobalWeighting global;
    private int[] cluster_sizes = new int[num_clusters];
    private Queue<OneClassClassifier> classifiers = new LinkedList<>();
    private static Remove filter = new Remove();

    public VOCL(VagueLabelMethod label_method) {
        this.label_method = label_method;
        this.local = new LocalWeighting(4); // Specify number of folds
        this.global = new GlobalWeighting();
        this.class_idx = 13; // FIXME
        this.attr_idx = 2; // FIXME
    }

    /**
     * This is the main VOCL method. This will apply vague one-class learning on the input stream.
     *
     * @param stream     The stream of instances
     * @param chunk_size The number of instances per chunk
     * @throws Exception
     */
    public void labelStream(Instances stream, final int chunk_size) throws Exception {
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

        // Does not process partial chunks
    }

    /**
     * Main Pseudo code method from paper (Figure 5.)
     *
     * @param chunk
     * @throws Exception
     */
    private void processChunk(Instances chunk) throws Exception {

        int[] PSi = vague_clustering.label(chunk);

        // Create copy: Filter chunk by attribute & specify chunk class index
        Instances attr_chunk = filterByAttribute(chunk, attr_idx);
        attr_chunk.setClassIndex(attr_chunk.numAttributes() - 1);

        float[] WLx = local.getWeights(attr_chunk, PSi, attr_idx, class_idx);
        float[] WGx = global.getWeights(attr_chunk, classifiers);

        float[] Wx = calculateUnifiedWeights(WLx, WGx);
        OneClassClassifier Li = trainNewClassifier(attr_chunk, Wx);

        // TODO weight classifiers
        if (classifiers.size() > 0) {
            float[] Gl = weightClassifiers(Li, Wx, attr_chunk, PSi);
        }
        // TODO form weighted classifier ensemble

        // Append new classifier to ensemble for make it k classifiers
        classifiers.add(Li);

        // Shift out oldest classifier
        if (classifiers.size() == k) classifiers.remove();
    }

    private float[] weightClassifiers(OneClassClassifier Li, float[] Wx, Instances chunk, int[] labels) throws Exception {

        float[] Gl = new float[classifiers.size()];

        Gl[classifiers.size() - 1] = pairWiseAgreement(Li, Li, chunk, labels);
        assert Float.compare(Gl[classifiers.size() - 1], 1.0f) == 0 : "pair-wise agreement against itself must be 1.";

        int i = 0;
        // Iterate from last recently used to most recently used (excludes Li since it hasn't been appended yet)
        for (OneClassClassifier classifier : classifiers) {
            Gl[i++] = pairWiseAgreement(Li, classifier, chunk, labels);
        }

        return Gl;
    }

    private float pairWiseAgreement(OneClassClassifier ol, OneClassClassifier oj, Instances chunk, int[] labels) throws Exception {

        int unlabeled_set_size = 0;
        float weight = 0.f;

        for (int i = 0; i < chunk.size(); i++) {
            // If unlabeled
            if (labels[i] == 0) {
                Instance instance = chunk.instance(i);
                // If classifiers predict same value increase weight
                if (Double.compare(ol.classifyInstance(instance), oj.classifyInstance(instance)) == 0) {
                    weight += 1.0f;
                }
                unlabeled_set_size++;
            }
        }

        return weight / unlabeled_set_size;
    }

    /**
     * Calculates unified weights using both Local and Global weights
     *
     * @param WLx The array of local weights
     * @param WGx The array of global weights
     * @return The array of unified weights
     */
    private float[] calculateUnifiedWeights(float[] WLx, float[] WGx) {

        assert WLx.length == WGx.length : "Weights have different lengths.";

        // Laplace smoothing parameters
        float a = 0.5f, b = 0.5f;

        float[] Wx = new float[WLx.length];

        for (int i = 0; i < Wx.length; i++) {
            if (WLx[i] + WGx[i] > 0) {
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
    private OneClassClassifier trainNewClassifier(Instances chunk, float[] unified_weights) throws Exception {

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

        // Train classifier with new weighted chunk
        new_classifier.buildClassifier(new_chunk);

        return new_classifier;
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
