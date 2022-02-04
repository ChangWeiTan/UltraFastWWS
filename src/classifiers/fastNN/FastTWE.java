package classifiers.fastNN;

import classifiers.classicNN.TWE1NN;
import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch
 * Code from "Efficient search of the best warping window for dynamic time warping"
 */
public class FastTWE extends TWE1NN {
    public FastTWE() {
        super();
        this.classifierIdentifier = "FastTWE";
    }

    public FastTWE(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "FastTWE";
    }

    public FastTWE(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "FastTWE";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
