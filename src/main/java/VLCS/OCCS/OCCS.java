package VLCS.OCCS;

import weka.clusterers.AbstractClusterer;
import weka.core.Instances;

public class OCCS {

    public OCCS() {

    }

    public void run(Instances chunks) {
        ClusterStep clusterStep = new ClusterStep();
        ConceptClusterMap conceptClusterMap = new ConceptClusterMap();
        try {
            AbstractClusterer clustering = clusterStep.run(chunks);
            conceptClusterMap.run(chunks,clustering);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
}
