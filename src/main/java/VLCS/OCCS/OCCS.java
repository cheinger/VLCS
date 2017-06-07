package VLCS.OCCS;

import weka.core.Instances;

public class OCCS {

    public OCCS() {

    }

    public void run(Instances chunks) {
        Clustering clustering = new Clustering();
        try {
            clustering.run(chunks);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
}
