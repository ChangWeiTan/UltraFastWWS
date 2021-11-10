package classifiers;

import application.Application;
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
public class UltraFastWWSearchFULL extends EAPDTW1NNLbKeogh {
    public UltraFastWWSearchFULL() {
        super();
        this.classifierIdentifier = "UltraFastWWSearchFull";
    }

    public UltraFastWWSearchFULL(final Sequences trainData) {
        super(trainData);
        this.classifierIdentifier = "UltraFastWWSearchFull";
    }

    public UltraFastWWSearchFULL(final int paramId, final Sequences trainData) {
        super(paramId, trainData);
        this.classifierIdentifier = "UltraFastWWSearchFull";
    }

    @Override
    public TrainingClassificationResults fastWWSearch(final Sequences train) throws Exception {
        bestParamId = -1;
        double bsfAcc = -1;
        double[] accAndPreds;
        this.maxWindow = train.length();
        int[] cvParams = new int[maxWindow + 1];
        double[] cvAcc = new double[maxWindow + 1];
        double[] predictions = new double[train.size()];

        if (Application.verbose > 1)
            System.out.print("[1-NN] Fast Parameter Search for " + this.classifierIdentifier + ", training ");

        final long start = System.nanoTime();
        if (Application.verbose > 1)
            System.out.println("[1-NN] Initialising NNs table for Fast Parameter Search");

        initNNSTable(train, trainCache);

        for (int win = 0; win < maxWindow + 1; win++) {
            cvParams[win] = win;
            if (Application.verbose > 1)
                System.out.print(".");
            accAndPreds = fastWWSearchAccAndPred(train, win, train.size());

            if (Application.verbose > 1)
                System.out.print(accAndPreds[0] + ",");
            cvAcc[win] = accAndPreds[0];
            if (accAndPreds[0] > bsfAcc) {
                bsfAcc = accAndPreds[0];
                bestParamId = win;
                System.arraycopy(accAndPreds, 1, predictions, 0, train.size());
            }
        }
        final long end = System.nanoTime();
        if (Application.verbose > 1)
            System.out.println("];");
        trainingTime = 1.0 * (end - start) / 1e9;

        final TrainingClassificationResults results = new TrainingClassificationResults(
                this.classifierIdentifier, bsfAcc, start, end, predictions);
        results.paramId = bestParamId;
        results.cvAcc = cvAcc;
        results.cvParams = cvParams;

        this.setTrainingData(train);
        this.setParamsFromParamId(bestParamId);
        if (Application.verbose > 1)
            System.out.printf("[1-NN] Fast Parameter Search Results: ParamID:=%d, %s, Acc=%.5f, Time=%s%n",
                    bestParamId, getParamInformationString(), bsfAcc, results.doTime());

        return results;
    }

    @Override
    protected double[] fastWWSearchAccAndPred(final Sequences train, final int paramId, final int n) {
        this.setParamsFromParamId(paramId, this.maxWindow);
        int correct = 0;
        double pred, actual;

        final double[] accAndPreds = new double[n + 1];
        for (int i = 0; i < n; i++) {
            actual = train.get(i).classificationLabel;
            pred = -1;
            double bsfCount = -1;
            for (int c = 0; c < classCounts[paramId][i].length; c++) {
                if (classCounts[paramId][i][c] > bsfCount) {
                    bsfCount = classCounts[paramId][i][c];
                    pred = c;
                }
            }
            if (pred == actual) {
                correct++;
            }
            accAndPreds[i + 1] = pred;
        }
        accAndPreds[0] = 1.0 * correct / n;

        return accAndPreds;
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
        final int maxWindow = train.length();
        double bestSoFar;

        candidateNNS = new CandidateNN[maxWindow + 1][train.size()];
        for (int window = 0; window < maxWindow + 1; ++window) {
            for (int len = 0; len < train.size(); ++len) {
                candidateNNS[window][len] = new CandidateNN();
            }
        }
        classCounts = new int[maxWindow + 1][train.size()][train.getNumClasses()];

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
            int win = maxWindow;

            // sort existing series by their distances to their NN
            final CandidateNN[] nnArr = candidateNNS[win];
            int[] sortedIndices = IntStream.range(0, current)
                    .boxed().sorted((i, j) -> nnArr[j].compareTo(nnArr[i]))
                    .mapToInt(ele -> ele).toArray();

            // find the NN at full DTW.
            CandidateNN currPNN = nnArr[current];
            for (int i = 0; i < sortedIndices.length; i++) {
                int sortedIdx = sortedIndices[i];
                AssessNNEAPDTW challenger = challengers.get(sortedIdx);
                final int previous = challenger.indexQuery;
                final CandidateNN prevNN = candidateNNS[win][previous];

                // --- First we want to beat the current best candidate:
                double toBeat = currPNN.distance;
                if (toBeat == Double.POSITIVE_INFINITY) {
                    challenger.tryEuclidean();
                    bestSoFar = challenger.euclideanDistance;
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
                        classCounts[win][current] = new int[train.getNumClasses()];
                        classCounts[win][current][challenger.getQuery().classificationLabel]++;
                    } else if (d == toBeat) {
                        classCounts[win][current][challenger.getQuery().classificationLabel]++;
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
                    prevNN.set(current, r, d, CandidateNN.Status.NN);
                    if (d < toBeat) {
                        classCounts[win][previous] = new int[train.getNumClasses()];
                        classCounts[win][previous][challenger.getReference().classificationLabel]++;
                    } else if (d == toBeat) {
                        classCounts[win][previous][challenger.getReference().classificationLabel]++;
                    }
                    for (int tmp = win - 1; tmp >= r; --tmp) {
                        candidateNNS[tmp][previous].set(current, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][previous] = classCounts[win][previous].clone();
                    }
                }
                maxWindowValidity = Math.max(maxWindowValidity, prevNN.r);
            }

            // --- When we looked at every past sequences,
            // the current best candidate is really the best one, so the NN.
            // So assign the current NN to all the windows that are valid
            int r = currPNN.r;
            double d = currPNN.distance;
            int index = currPNN.nnIndex;
            maxWindowValidity = Math.max(maxWindowValidity, r);
            for (int tmp = win; tmp >= r; --tmp) {
                candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                classCounts[tmp][current] = classCounts[win][current].clone();
            }

            // now sort the existing series based on distance at w+1
            Collections.sort(challengers);

            // remember the NN at w+1
            int nnAtPreviousWindow = 0;

            for (win = maxWindowValidity - 1; win > -1; --win) {
                currPNN = candidateNNS[win][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int i = 0; i < current; ++i) {
                        final AssessNNEAPDTW challenger = challengers.get(i);
                        int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[win][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        if (toBeat == Double.POSITIVE_INFINITY) {
                            challenger.tryEuclidean();
                            bestSoFar = challenger.euclideanDistance;
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
                                classCounts[win][previous] = new int[train.getNumClasses()];
                                classCounts[win][previous][challenger.getReference().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[win][previous][challenger.getReference().classificationLabel]++;
                            }
                        }
                    }
                } else {
                    // --- --- WITHOUT NN CASE --- ---
                    // We don't have the NN yet.
                    AssessNNEAPDTW challenger = challengers.get(nnAtPreviousWindow);
                    int previous = challenger.indexQuery;
                    CandidateNN prevNN = candidateNNS[win][previous];

                    // --- First we want to beat the current best candidate:
                    double toBeat = currPNN.distance;
                    if (toBeat == Double.POSITIVE_INFINITY) {
                        challenger.tryEuclidean();
                        bestSoFar = challenger.euclideanDistance;
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
                            classCounts[win][current] = new int[train.getNumClasses()];
                            classCounts[win][current][challenger.getQuery().classificationLabel]++;
                        } else if (d == toBeat) {
                            classCounts[win][current][challenger.getQuery().classificationLabel]++;
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
                            classCounts[win][previous] = new int[train.getNumClasses()];
                            classCounts[win][previous][challenger.getReference().classificationLabel]++;
                        } else if (d == toBeat) {
                            classCounts[win][previous][challenger.getReference().classificationLabel]++;
                        }
                    }

                    for (int i = 0; i < current; ++i) {
                        // skip the NN
                        if (i == nnAtPreviousWindow)
                            continue;

                        challenger = challengers.get(i);
                        previous = challenger.indexQuery;
                        prevNN = candidateNNS[win][previous];

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
                        if (rrt == AssessNNEAPDTW.RefineReturnType.New_best) {
                            r = challenger.getMinWindowValidityForFullDistance();
                            d = challenger.getDistance(win);
                            currPNN.set(previous, r, d, CandidateNN.Status.BC);
                            if (d < toBeat) {
                                nnAtPreviousWindow = i;
                                classCounts[win][current] = new int[train.getNumClasses()];
                                classCounts[win][current][challenger.getQuery().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[win][current][challenger.getQuery().classificationLabel]++;
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
                                classCounts[win][previous] = new int[train.getNumClasses()];
                                classCounts[win][previous][challenger.getReference().classificationLabel]++;
                            } else if (d == toBeat) {
                                classCounts[win][previous][challenger.getReference().classificationLabel]++;
                            }
                        }
                    }

                    // --- When we looked at every past sequences,
                    // the current best candidate is really the best one, so the NN.
                    // So assign the current NN to all the windows that are valid
                    r = currPNN.r;
                    d = currPNN.distance;
                    index = currPNN.nnIndex;
                    for (int tmp = win; tmp >= r; --tmp) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[win][current].clone();
                    }
                }
            }
        }
    }

    public void setParamsFromParamId(final int win, final int maxWindow) {
        if (win < 0) return;

        if (win < maxWindow && this.classifierIdentifier.contains("R1")) {
            this.classifierIdentifier = this.classifierIdentifier.replace("R1", "Rn");
        }
        r = 1.0 * win / maxWindow;
        window = win;
    }
}
