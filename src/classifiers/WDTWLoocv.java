package classifiers;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * WDTW-1NN with LOOCV training
 */
public class WDTWLoocv extends WDTW1NN {
    public WDTWLoocv() {
        super();
        this.classifierIdentifier = "WDTW_1NN-LOOCV";
    }

    public WDTWLoocv(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "WDTW_1NN-LOOCV";
    }

    public WDTWLoocv(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "WDTW_1NN-LOOCV";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv(this.trainData);
    }
}
