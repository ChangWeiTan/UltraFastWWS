package classifiers;

import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch
 * Code from "Efficient search of the best warping window for dynamic time warping"
 */
public class FastWDTW extends WDTW1NN {
    public FastWDTW() {
        super();
        this.classifierIdentifier = "FastWDTW";
    }

    public FastWDTW(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "FastWDTW";
    }

    public FastWDTW(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "FastWDTW";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
