package classifiers.fastNN;

import classifiers.classicNN.ERP1NN;
import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch
 * Code from "Efficient search of the best warping window for dynamic time warping"
 */
public class FastERP extends ERP1NN {
    public FastERP() {
        super();
        this.classifierIdentifier = "FastERP";
    }

    public FastERP(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "FastERP";
    }

    public FastERP(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "FastERP";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
