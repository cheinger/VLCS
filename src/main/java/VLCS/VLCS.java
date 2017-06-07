package VLCS;

import VLCS.OCCS.OCCS;
import VLCS.VOCL.VOCL;
import weka.core.converters.ArffLoader.ArffReader;
import moa.clusterers.kmeanspm.ClusteringFeature;
/*
 *  In order to address the vague one-class learning and concept summarization challenges, we propose a VLCS system with
 *  two major modules. Given a data stream with samples arriving in a chunk-by-chunk manner, VLCS
 *  achieves vague one-class learning by using the VOCL module, and the concept summarization is achieved through the
 *  OCCS module.
 */
public class VLCS {
    ArffReader arff;

    public VLCS(ArffReader arff_){
        arff = arff_;
    }
    public void run(){
        VOCL vocl = new VOCL(VOCL.VagueLabelMethod.CLUSTER, 1);
        OCCS occs = new OCCS();

    }
}
