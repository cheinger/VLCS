package VLCS.OCCS;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

/**
 * clustering based techniques to merge data chunks into a number of
 * groups, each of which denotes a concept
 */
public class Clustering {

    SimpleKMeans simpleKMeans;

    Clustering(){

    }

    public Cluster run(Instances chunks) throws Exception {
        // (1) extract features from each chunk (i.e., set) and
        // represent all chunks as a virtual sample set
        VirtualSampleSet virtualSet = FeatureExtraction.fromChunks(chunks);

        // (2) generate cluster structures from the virtual set.
        simpleKMeans = new SimpleKMeans();
        simpleKMeans.setNumClusters(16);
        simpleKMeans.buildClusterer(chunks);

        return new Cluster();
    }

    class Concept {
        String concept;
    }

    class Cluster {
        Concept concept;
    }
}
