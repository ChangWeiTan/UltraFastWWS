package classifiers.classicNN;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * LCSS-1NN with LOOCV training
 */
public class LCSSLoocv extends LCSS1NN {
    public LCSSLoocv() {
        super();
        this.classifierIdentifier = "LCSS_1NN-LOOCV";
    }

    public LCSSLoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "LCSS_1NN-LOOCV";
    }

    public LCSSLoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "LCSS_1NN-LOOCV";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv(this.trainData);
    }
}
