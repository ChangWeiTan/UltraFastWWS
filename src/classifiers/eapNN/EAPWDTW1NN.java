package classifiers.eapNN;

import classifiers.classicNN.OneNearestNeighbour;
import datasets.Sequence;
import datasets.Sequences;
import distances.eap.EAPWDTW;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.LazyAssessNNEAPWDTW;
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
        if (train.size() < 2) {
            System.err.println("[INIT-NNS-TABLE] Set is too small: " + train.size() + " sequence. At least 2 sequences needed.");
        }

        candidateNNS = new CandidateNN[nParams][train.size()];
        for (int paramId = 0; paramId < nParams; ++paramId) {
            for (int len = 0; len < train.size(); ++len) {
                candidateNNS[paramId][len] = new CandidateNN();
            }
        }
        classCounts = new int[nParams][train.size()][train.getNumClasses()];
        boolean[] vectorCreated = new boolean[nParams];
        double[][] weightVectors = new double[nParams][maxWindow];

        final LazyAssessNNEAPWDTW[] lazyAssessNNS = new LazyAssessNNEAPWDTW[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNNEAPWDTW(cache);
        }
        final ArrayList<LazyAssessNNEAPWDTW> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNNEAPWDTW d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(paramId);
                if (!vectorCreated[paramId]) {
                    initWeights(sCurrent.length());
                    weightVectors[paramId] = weightVector;
                    vectorCreated[paramId] = true;
                }
                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNNEAPWDTW challenger = lazyAssessNNS[previous];
                        final LazyAssessNNEAPWDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, weightVectors[paramId]);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPWDTW.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            prevNN.set(current, d, CandidateNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.getNumClasses()];
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            }
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have the NN yet.
                    // Sort the challengers so we have the better chance to organize the good pruning.
                    Collections.sort(challengers);

                    for (LazyAssessNNEAPWDTW challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNNEAPWDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, weightVectors[paramId]);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPWDTW.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            currPNN.set(previous, d, CandidateNN.Status.BC);
                            if (d < toBeat) {
                                classCounts[paramId][current] = new int[train.getNumClasses()];
                                classCounts[paramId][current][challenger.getQuery().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][current][challenger.getQuery().classificationLabel]++;
                            }
                        }

                        // --- Now check for previous NN
                        // --- Try to beat the previous best NN
                        toBeat = prevNN.distance;
                        challenger = lazyAssessNNS[previous];
                        rrt = challenger.tryToBeat(toBeat, weightVectors[paramId]);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPWDTW.RefineReturnType.New_best) {
                            final double d = challenger.getDistance();
                            prevNN.set(current, d, CandidateNN.Status.NN);
                            if (d < toBeat) {
                                classCounts[paramId][previous] = new int[train.getNumClasses()];
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                            }
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    candidateNNS[paramId][current].set(currPNN.nnIndex, currPNN.distance, CandidateNN.Status.NN);
                    classCounts[paramId][current] = classCounts[paramId][current].clone();
                }
            }
        }
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
