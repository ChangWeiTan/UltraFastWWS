package classifiers.fastNN;

import classifiers.classicNN.LCSS1NN;
import classifiers.classicNN.TWE1NN;
import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch
 * Code from "Efficient search of the best warping window for dynamic time warping"
 */
public class FastLCSS extends LCSS1NN {
    public FastLCSS() {
        super();
        this.classifierIdentifier = "FastLCSS";
    }

    public FastLCSS(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "FastLCSS";
    }

    public FastLCSS(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "FastLCSS";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
