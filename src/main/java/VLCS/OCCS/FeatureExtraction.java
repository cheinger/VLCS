package VLCS.OCCS;

import weka.core.Instances;

public class FeatureExtraction {
    // To extract
    // feature for each chunk, we propose to transform original feature
    // values in each chunk into some histogram format, such that each
    // chunk Si can be represented by using histogram based features
    public static VirtualSampleSet fromChunks(Instances chunks) {
        return new VirtualSampleSet();
    }
}
