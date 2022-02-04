package classifiers.ultraFastNN;

import classifiers.eapNN.EAPDTW1NNLbKeogh;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.lazyAssessNNEAP.LazyAssessNNEAPDTW;
import results.TrainingClassificationResults;

import java.util.ArrayList;
import java.util.Collections;
import java.util.stream.IntStream;

/**
 * UltraFastWWSearch (V2)
 * "Ultra fast warping window optimization for Dynamic Time Warping"
 * EAPFastWWSearchEA without lower bound
 */
public class UltraFastWWSearchV2 extends EAPDTW1NNLbKeogh {
    public UltraFastWWSearchV2() {
        super();
        this.classifierIdentifier = "UltraFastWWSearchV2";
    }

    public UltraFastWWSearchV2(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "UltraFastWWSearchV2";
    }

    public UltraFastWWSearchV2(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "UltraFastWWSearchV2";
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

        // initialise
        double bestSoFar;

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

            // ============================================================
            // do the largest window (full DTW) first
            // get paramid and initialise
            int paramId = nParams - 1;
            setParamsFromParamId(paramId);
            int win = this.window;

            // sort existing series by their distances to their NN
            final CandidateNN[] nnArr = candidateNNS[paramId];
            int[] sortedIndices = IntStream.range(0, current)
                    .boxed().sorted((i, j) -> nnArr[j].compareTo(nnArr[i]))
                    .mapToInt(ele -> ele).toArray();

            // find the NN at full DTW.
            CandidateNN currPNN = nnArr[current];
            for (int i = 0; i < sortedIndices.length; i++) {
                int sortedIdx = sortedIndices[i];
                LazyAssessNNEAPDTW challenger = challengers.get(sortedIdx);
                final int previous = challenger.indexQuery;
                final CandidateNN prevNN = candidateNNS[paramId][previous];

                // --- First we want to beat the current best candidate:
                double toBeat = currPNN.distance;
                if (toBeat == Double.POSITIVE_INFINITY) {
                    challenger.tryEuclidean();
                    bestSoFar = challenger.euclideanDistance;
                } else {
                    bestSoFar = Math.max(toBeat, prevNN.distance);
                }
                LazyAssessNNEAPDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win, bestSoFar);

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
                rrt = challenger.tryToBeat(toBeat, win, bestSoFar);

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
            int r = currPNN.r;
            double d = currPNN.distance;
            int index = currPNN.nnIndex;
            int winEnd = getParamIdFromWindow(r, train.length());
            for (int tmp = paramId; tmp >= winEnd; --tmp) {
                candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                classCounts[tmp][current] = classCounts[paramId][current].clone();
            }

            // now sort the existing series based on distance at w+1
            Collections.sort(challengers);

            // remember the NN at w+1
            int nnAtPreviousWindow = 0;

            // go through the parameters to fill the NNS table
            for (paramId = nParams - 2; paramId > -1; --paramId) {
                setParamsFromParamId(paramId);
                win = distComputer.getWindowSize(maxWindow, this.r);
                currPNN = candidateNNS[paramId][current]; // get the current NN

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is
                    // the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNNEAPDTW challenger = lazyAssessNNS[previous];
                        if (toBeat == Double.POSITIVE_INFINITY) {
                            challenger.tryEuclidean();
                            bestSoFar = challenger.euclideanDistance;
                        } else {
                            bestSoFar = toBeat;
                        }
                        final LazyAssessNNEAPDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win, bestSoFar);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPDTW.RefineReturnType.New_best) {
                            r = challenger.getMinWindowValidityForFullDistance();
                            d = challenger.getDistance(win);
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
                    LazyAssessNNEAPDTW challenger = challengers.get(nnAtPreviousWindow);
                    int previous = challenger.indexQuery;
                    CandidateNN prevNN = candidateNNS[paramId][previous];

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    if (toBeat == Double.POSITIVE_INFINITY) {
                        challenger.tryEuclidean();
                        bestSoFar = challenger.euclideanDistance;
                    } else {
                        bestSoFar = Math.max(toBeat, prevNN.distance);
                    }
                    LazyAssessNNEAPDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win, bestSoFar);

                    // --- Check the result
                    if (rrt == LazyAssessNNEAPDTW.RefineReturnType.New_best) {
                        r = challenger.getMinWindowValidityForFullDistance();
                        d = challenger.getDistance(win);
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
                    rrt = challenger.tryToBeat(toBeat, win, bestSoFar);

                    // --- Check the result
                    if (rrt == LazyAssessNNEAPDTW.RefineReturnType.New_best) {
                        r = challenger.getMinWindowValidityForFullDistance();
                        d = challenger.getDistance(win);
                        prevNN.set(current, r, d, CandidateNN.Status.NN);
                        if (d < toBeat) {
                            classCounts[paramId][previous] = new int[train.getNumClasses()];
                            classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                        } else if (d == toBeat) {
                            classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                        }
                    }

                    for (int i = 0; i < current; ++i) {
                        // skip the NN
                        if (i == nnAtPreviousWindow)
                            continue;

                        challenger = challengers.get(i);
                        previous = challenger.indexQuery;
                        prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        toBeat = currPNN.distance;
                        if (toBeat == Double.POSITIVE_INFINITY) {
                            challenger.tryEuclidean();
                            bestSoFar = challenger.euclideanDistance;
                        } else {
                            bestSoFar = Math.max(toBeat, prevNN.distance);
                        }
                        rrt = challenger.tryToBeat(toBeat, win, bestSoFar);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPDTW.RefineReturnType.New_best) {
                            r = challenger.getMinWindowValidityForFullDistance();
                            d = challenger.getDistance(win);
                            currPNN.set(previous, r, d, CandidateNN.Status.BC);
                            if (d < toBeat) {
                                nnAtPreviousWindow = i;
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
                        rrt = challenger.tryToBeat(toBeat, win, bestSoFar);

                        // --- Check the result
                        if (rrt == LazyAssessNNEAPDTW.RefineReturnType.New_best) {
                            r = challenger.getMinWindowValidityForFullDistance();
                            d = challenger.getDistance(win);
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
                    r = currPNN.r;
                    d = currPNN.distance;
                    index = currPNN.nnIndex;
                    winEnd = getParamIdFromWindow(r, train.length());
                    for (int tmp = paramId; tmp >= winEnd; --tmp) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();
                    }
                }
            }
        }
    }
}
