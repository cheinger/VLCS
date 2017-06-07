package VLCS.OCCS;

import weka.clusterers.AbstractClusterer;
import weka.clusterers.Cobweb;
import weka.core.Instances;

/**
 * clustering based techniques to merge data chunks into a number of
 * groups, each of which denotes a concept
 */
public class ClusterStep {

    private Cobweb clusterer;

    ClusterStep() {
        clusterer = new Cobweb();
    }

    public AbstractClusterer run(Instances chunks) throws Exception {

//        clusterer.resetLearningImpl();
//
//        for (Instance chunk : chunks) {
//            clusterer.trainOnInstanceImpl(chunk);
//        }
//
//        return clusterer.getClusteringResult();

        clusterer.buildClusterer(chunks);

//        clusterer.clusterInstance()
//        clusterer.distributionForInstance()

        return clusterer;
    }
}
