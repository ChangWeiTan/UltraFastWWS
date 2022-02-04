package classifiers.fastNN;

import classifiers.classicNN.DTW1NNLbKeogh;
import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch
 * Code from "Efficient search of the best warping window for dynamic time warping"
 */
public class FastWWSearch extends DTW1NNLbKeogh {
    public FastWWSearch() {
        super();
        this.classifierIdentifier = "FastWWSearch-LbKeogh";
    }

    public FastWWSearch(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "FastWWSearch-LbKeogh";
    }

    public FastWWSearch(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "FastWWSearch-LbKeogh";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
