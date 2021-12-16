package classifiers;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch
 * Code from "Efficient search of the best warping window for dynamic time warping"
 */
public class FastMSM extends MSM1NN {
    public FastMSM() {
        super();
        this.classifierIdentifier = "FastMSM";
    }

    public FastMSM(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "FastMSM";
    }

    public FastMSM(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "FastMSM";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
