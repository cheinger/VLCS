package VLCS.OCCS;

import weka.clusterers.AbstractClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

/**
 * The concept cluster map directly employs the cluster structure to
 * visually demonstrate concept-chunk relationships in the stream
 */
public class ConceptClusterMap {

    SimpleKMeans simpleKMeans;

    ConceptClusterMap() {
    }

    public int[] run(Instances chunks, AbstractClusterer clustering) throws Exception {

//        ArrayList<Cluster> clusters = clustering.getClustering();
//
//        clusters.stream().forEach(c -> System.out.println(c.getInfo()));
//        Double[] weights = clusters.stream().map((c) -> c.getWeight()).toArray(Double[]::new);
//
//        System.out.println(weights);

        simpleKMeans = new SimpleKMeans();
        simpleKMeans.setNumClusters(16);
        simpleKMeans.buildClusterer(chunks);

        int[] weights = simpleKMeans.getAssignments();
        assert chunks.size() == weights.length;
        return weights;
    }
}
