package classifiers.eapNN;

import classifiers.classicNN.DTW1NN;
import datasets.Sequence;
import datasets.Sequences;
import distances.eap.EAPDTW;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.lazyAssessNNEAP.LazyAssessNNEAPDTW;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Super class for EAPDTW-1NN
 * EAPDTW-1NN with no lower bounds
 */
public class EAPDTW1NN extends DTW1NN {
    protected EAPDTW distComputer = new EAPDTW();

    public EAPDTW1NN() {
        this.classifierIdentifier = "EAPDTW_1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public EAPDTW1NN(final Sequences trainData) {
        this.setTrainingData(trainData);
        this.r = 1;
        this.window = trainData.length();
        this.classifierIdentifier = "EAPDTW_1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public EAPDTW1NN(final int paramId, final Sequences trainData) {
        this.r = 1;
        this.window = trainData.get(0).length();
        this.classifierIdentifier = "EAPDTW_1NN_R1";
        this.setTrainingData(trainData);
        this.setParamsFromParamId(paramId);
        this.bestParamId = paramId;
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    @Override
    public double distance(final Sequence first, final Sequence second) {
        if (r < 1) {
            window = distComputer.getWindowSize(Math.max(first.length(), second.length()), r);
            return distComputer.distance(first.data[0], second.data[0], window, Double.POSITIVE_INFINITY);
        }
        return distComputer.distance(first.data[0], second.data[0], Double.POSITIVE_INFINITY);
    }

    @Override
    public double distance(final Sequence first, final Sequence second, final double cutOffValue) {
        if (r < 1) {
            window = distComputer.getWindowSize(Math.max(first.length(), second.length()), r);
            return distComputer.distance(first.data[0], second.data[0], window, cutOffValue);
        }
        return distComputer.distance(first.data[0], second.data[0], cutOffValue);
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

        final LazyAssessNNEAPDTW[] lazyAssessNNS = new LazyAssessNNEAPDTW[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNNEAPDTW(cache);
        }
        final ArrayList<LazyAssessNNEAPDTW> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            // get all the instances from 0 -> current
            // these form the "temporary" training set
            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNNEAPDTW d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            // go through the parameters to fill the NNS table
            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(paramId);
                final int win = distComputer.getWindowSize(maxWindow, r);
                final CandidateNN currPNN = candidateNNS[paramId][current]; // get the current NN

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is
                    // the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNNEAPDTW challenger = lazyAssessNNS[previous];
                        final LazyAssessNNEAPDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win, Double.POSITIVE_INFINITY);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPDTW.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(win);
                            prevNN.set(current, r, d, CandidateNN.Status.NN);
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

                    for (LazyAssessNNEAPDTW challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNNEAPDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win, Double.POSITIVE_INFINITY);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPDTW.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(win);
                            currPNN.set(previous, r, d, CandidateNN.Status.BC);
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
                        rrt = challenger.tryToBeat(toBeat, win, Double.POSITIVE_INFINITY);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPDTW.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(win);
                            prevNN.set(current, r, d, CandidateNN.Status.NN);
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
                    final int r = currPNN.r;
                    final double d = currPNN.distance;
                    final int index = currPNN.nnIndex;
                    final int winEnd = getParamIdFromWindow(r, train.length());
                    for (int tmp = paramId; tmp >= winEnd; --tmp) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();
                    }
                }
            }
        }
    }
}
