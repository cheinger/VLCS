package VLCS.VOCL;

import weka.core.Instances;

/**
 * Created by xavier on 7/06/17.
 */
public abstract class VagueLabeling {
    protected float alpha = 0.3f;

    public abstract int[] label(Instances chunk) throws Exception;
}
