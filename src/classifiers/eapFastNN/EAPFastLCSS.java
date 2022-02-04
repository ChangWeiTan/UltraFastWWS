package classifiers.eapFastNN;

import classifiers.classicNN.LCSS1NN;
import classifiers.eapNN.EAPLCSS1NN;
import datasets.Sequences;
import results.TrainingClassificationResults;

/**
 * FastWWSearch
 * Code from "Efficient search of the best warping window for dynamic time warping"
 */
public class EAPFastLCSS extends EAPLCSS1NN {
    public EAPFastLCSS() {
        super();
        this.classifierIdentifier = "EAPFastLCSS";
    }

    public EAPFastLCSS(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "EAPFastLCSS";
    }

    public EAPFastLCSS(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "EAPFastLCSS";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }
}
