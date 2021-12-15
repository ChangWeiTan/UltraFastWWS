package classifiers;

import datasets.Sequence;
import datasets.Sequences;
import distances.classic.WDTW;
import distances.eap.EAPDTW;
import distances.eap.EAPWDTW;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.LazyAssessNNEAPDTW;
import results.TrainingClassificationResults;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Super class for EAPDTW-1NN
 * EAPDTW-1NN with no lower bounds
 */
public class EAPWDTW1NN extends OneNearestNeighbour {
    protected static final double WEIGHT_MAX = 1;
    protected double g = 0;                    // g value
    protected double[] weightVector;           // weights vector
    protected boolean refreshWeights = true;   // indicator if we refresh the params
    protected EAPWDTW distComputer = new EAPWDTW();

    protected void initWeights(int seriesLength) {
        weightVector = new double[seriesLength];
        double halfLength = (double) seriesLength / 2;

        for (int i = 0; i < seriesLength; i++) {
            weightVector[i] = WEIGHT_MAX / (1 + Math.exp(-g * (i - halfLength)));
        }
        refreshWeights = false;
    }

    public EAPWDTW1NN() {
        this.classifierIdentifier = "EAPWDTW_1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public EAPWDTW1NN(final Sequences trainData) {
        this.setTrainingData(trainData);
        initWeights(trainData.length());
        this.classifierIdentifier = "EAPWDTW_1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public EAPWDTW1NN(final int paramId, final Sequences trainData) {
        this.classifierIdentifier = "EAPWDTW_1NN_R1";
        this.setTrainingData(trainData);
        g = (double) paramId / 100;
        refreshWeights = true;
        initWeights(trainData.length());
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public void summary() {
        System.out.println(toString());
    }

    @Override
    public String toString() {
        return "[CLASSIFIER SUMMARY] Classifier: " + this.classifierIdentifier +
                "\n[CLASSIFIER SUMMARY] training_opts: " + trainingOptions +
                "\n[CLASSIFIER SUMMARY] g: " + g +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final Sequence first, final Sequence second) {
        return distComputer.distance(first.data[0], second.data[0], weightVector, Double.POSITIVE_INFINITY);
    }

    @Override
    public double distance(final Sequence first, final Sequence second, final double cutOffValue) {
        return distComputer.distance(first.data[0], second.data[0], weightVector, cutOffValue);
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return loocv0(this.trainData);
    }

    /**
     * Code from "Efficient search of the best warping window for dynamic time warping"
     */
    @Override
    public void initNNSTable(final Sequences train, final SequenceStatsCache cache) {

    }

    @Override
    public void setTrainingData(final Sequences trainData) {
        this.trainData = trainData;
        this.trainCache = new SequenceStatsCache(trainData, trainData.get(0).length());
    }

    @Override
    public void setParamsFromParamId(final int paramId) {
        if (paramId < 0) return;

        if (paramId < 100 && this.classifierIdentifier.contains("R1")) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
        }
        g = (double) paramId / 100;
        refreshWeights = true;
        initWeights(trainData.length());
    }

    @Override
    public String getParamInformationString() {
        return "g=" + this.g;
    }
}
