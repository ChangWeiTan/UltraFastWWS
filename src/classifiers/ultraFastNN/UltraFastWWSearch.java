package classifiers.ultraFastNN;

import classifiers.eapNN.EAPDTW1NNLbKeogh;
import datasets.Sequence;
import datasets.Sequences;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.assessNN.AssessNNEAPDTW;
import results.TrainingClassificationResults;

import java.util.ArrayList;
import java.util.Collections;
import java.util.stream.IntStream;

/**
 * UltraFastWWSearch
 * "Ultra fast warping window optimization for Dynamic Time Warping"
 */
public class UltraFastWWSearch extends EAPDTW1NNLbKeogh {
    final int ubParam = 0;

    public UltraFastWWSearch() {
        super();
        this.classifierIdentifier = "UltraFastWWSearch";
    }

    public UltraFastWWSearch(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "UltraFastWWSearch";
    }

    public UltraFastWWSearch(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "UltraFastWWSearch";
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

        final AssessNNEAPDTW[] lazyAssessNNS = new AssessNNEAPDTW[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new AssessNNEAPDTW(cache);
        }
        final ArrayList<AssessNNEAPDTW> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            int maxWindowValidity = 0;
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final AssessNNEAPDTW d = lazyAssessNNS[previous];
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
                AssessNNEAPDTW challenger = challengers.get(sortedIdx);
                final int previous = challenger.indexQuery;
                final CandidateNN prevNN = candidateNNS[paramId][previous];

                // --- First we want to beat the current best candidate:
                double toBeat = currPNN.distance;
                if (toBeat == Double.POSITIVE_INFINITY) {
                    if (candidateNNS[ubParam][current].distance == Double.POSITIVE_INFINITY) {
                        challenger.tryEuclidean();
                        bestSoFar = challenger.euclideanDistance;
                        candidateNNS[ubParam][current].set(previous, 0, bestSoFar, CandidateNN.Status.BC);
                    } else {
                        bestSoFar = candidateNNS[ubParam][current].distance;
                    }
                } else {
                    bestSoFar = Math.max(toBeat, prevNN.distance);
                }
                AssessNNEAPDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win, bestSoFar);

                // --- Check the result
                if (rrt == AssessNNEAPDTW.RefineReturnType.New_best) {
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
                if (rrt == AssessNNEAPDTW.RefineReturnType.New_best) {
                    final int r = challenger.getMinWindowValidityForFullDistance();
                    final double d = challenger.getDistance(win);
                    final int winEnd = getParamIdFromWindow(r, train.length());
                    prevNN.set(current, r, d, CandidateNN.Status.NN);
                    if (d < toBeat) {
                        classCounts[paramId][previous] = new int[train.getNumClasses()];
                        classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                    } else if (d == toBeat) {
                        classCounts[paramId][previous][challenger.getReference().classificationLabel]++;
                    }
                    for (int tmp = paramId - 1; tmp >= winEnd; --tmp) {
                        candidateNNS[tmp][previous].set(current, r, winEnd, d, CandidateNN.Status.NN);
                        classCounts[tmp][previous] = classCounts[paramId][previous].clone();
                    }
                }
                int winEnd = getParamIdFromWindow(prevNN.r, train.length());
                maxWindowValidity = Math.max(maxWindowValidity, winEnd);
            }

            // --- When we looked at every past sequences,
            // the current best candidate is really the best one, so the NN.
            // So assign the current NN to all the windows that are valid
            int r = currPNN.r;
            double d = currPNN.distance;
            int index = currPNN.nnIndex;
            int winEnd = getParamIdFromWindow(r, train.length());
            maxWindowValidity = Math.max(maxWindowValidity, winEnd);
            for (int tmp = paramId; tmp >= winEnd; --tmp) {
                candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                classCounts[tmp][current] = classCounts[paramId][current].clone();
            }

            // now sort the existing series based on distance at w+1
            Collections.sort(challengers);

            // remember the NN at w+1
            int nnAtPreviousWindow = 0;

            for (paramId = maxWindowValidity - 1; paramId > -1; --paramId) {
                setParamsFromParamId(paramId);
                win = distComputer.getWindowSize(maxWindow, this.r);
                currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int i = 0; i < current; ++i) {
                        final AssessNNEAPDTW challenger = challengers.get(i);
                        int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        if (toBeat == Double.POSITIVE_INFINITY) {
                            if (candidateNNS[ubParam][previous].distance == Double.POSITIVE_INFINITY) {
                                challenger.tryEuclidean();
                                bestSoFar = challenger.euclideanDistance;
                                candidateNNS[ubParam][previous].set(current, 0, bestSoFar, CandidateNN.Status.BC);
                            } else {
                                bestSoFar = candidateNNS[ubParam][previous].distance;
                            }
                        } else {
                            bestSoFar = toBeat;
                        }
                        final AssessNNEAPDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win, bestSoFar);

                        // --- Check the result
                        if (rrt == AssessNNEAPDTW.RefineReturnType.New_best) {
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
                    AssessNNEAPDTW challenger = challengers.get(nnAtPreviousWindow);
                    int previous = challenger.indexQuery;
                    CandidateNN prevNN = candidateNNS[paramId][previous];

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    if (toBeat == Double.POSITIVE_INFINITY) {
                        if (candidateNNS[ubParam][current].distance == Double.POSITIVE_INFINITY) {
                            challenger.tryEuclidean();
                            bestSoFar = challenger.euclideanDistance;
                            candidateNNS[ubParam][current].set(previous, 0, bestSoFar, CandidateNN.Status.BC);
                        } else {
                            bestSoFar = candidateNNS[ubParam][current].distance;
                        }
                    } else {
                        bestSoFar = Math.max(toBeat, prevNN.distance);
                    }
                    AssessNNEAPDTW.RefineReturnType rrt = challenger.tryToBeat(toBeat, win, bestSoFar);

                    // --- Check the result
                    if (rrt == AssessNNEAPDTW.RefineReturnType.New_best) {
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
                    if (rrt == AssessNNEAPDTW.RefineReturnType.New_best) {
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
                            if (candidateNNS[ubParam][current].distance == Double.POSITIVE_INFINITY) {
                                challenger.tryEuclidean();
                                bestSoFar = challenger.euclideanDistance;
                                candidateNNS[ubParam][current].set(previous, 0, bestSoFar, CandidateNN.Status.BC);
                            } else {
                                bestSoFar = candidateNNS[ubParam][current].distance;
                            }
                        } else {
                            bestSoFar = Math.max(toBeat, prevNN.distance);
                        }
                        rrt = challenger.tryToBeat(toBeat, win, bestSoFar);

                        // --- Check the result
                        if (rrt == AssessNNEAPDTW.RefineReturnType.New_best) {
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
                        if (rrt == AssessNNEAPDTW.RefineReturnType.New_best) {
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
