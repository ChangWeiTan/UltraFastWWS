package classifiers;

import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.LazyAssessNNDTW_nolb;
import results.TrainingClassificationResults;

import java.util.ArrayList;
import java.util.Collections;

/**
 * DTW-1NN with FastCV training and Lb Keogh
 */
public class FastWWSearchNoLb extends DTW1NNLbKeogh {
    public FastWWSearchNoLb() {
        super();
        this.classifierIdentifier = "DTW_1NN-FastCV_NoLb-LbKeogh";
    }

    public FastWWSearchNoLb(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "DTW_1NN-FastCV_NoLb-LbKeogh";
    }

    public FastWWSearchNoLb(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "DTW_1NN-FastCV_NoLb-LbKeogh";
    }

    @Override
    public TrainingClassificationResults fit(final Sequences trainData) throws Exception {
        this.setTrainingData(trainData);
        return fastWWSearch(this.trainData);
    }

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

        final LazyAssessNNDTW_nolb[] lazyAssessNNS = new LazyAssessNNDTW_nolb[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNNDTW_nolb(cache);
        }
        final ArrayList<LazyAssessNNDTW_nolb> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNNDTW_nolb d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(paramId);
                final int win = distComputer.getWindowSize(maxWindow, r);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNNDTW_nolb challenger = lazyAssessNNS[previous];
                        final LazyAssessNNDTW_nolb.RefineReturnType rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNNDTW_nolb.RefineReturnType.New_best) {
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

                    for (LazyAssessNNDTW_nolb challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNNDTW_nolb.RefineReturnType rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNNDTW_nolb.RefineReturnType.New_best) {
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
                        rrt = challenger.tryToBeat(toBeat, win);

                        // --- Check the result
                        if (rrt == LazyAssessNNDTW_nolb.RefineReturnType.New_best) {
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
