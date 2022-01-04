package classifiers.classicNN;

import datasets.Sequence;
import datasets.Sequences;
import distances.classic.ERP;
import fastWWS.CandidateNN;
import fastWWS.SequenceStatsCache;
import fastWWS.lazyAssessNN.LazyAssessNNERP;
import results.TrainingClassificationResults;
import utils.GenericTools;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Super class for ERP-1NN
 * MSM-1NN with no lower bounds
 */
public class ERP1NN extends OneNearestNeighbour {
    // parameters
    protected double g;                               // g value
    protected double bandSize;                        // band size in terms of percentage of sequence's length
    protected int window;

    protected double[] gValues;                       // set of g values
    protected double[] bandSizes;                     // set of band sizes
    protected boolean gAndWindowsRefreshed = false;   // indicator if we refresh the params

    protected ERP distComputer = new ERP();

    public ERP1NN() {
        this.classifierIdentifier = "ERP-1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public ERP1NN(final Sequences trainData) {
        this.setTrainingData(trainData);
        this.classifierIdentifier = "ERP-1NN_R1";
        this.trainingOptions = TrainOpts.LOOCV0;
    }

    public ERP1NN(final int paramId, final Sequences trainData) {
        this.classifierIdentifier = "ERP-1NN_R1";
        this.setTrainingData(trainData);
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
                "\n[CLASSIFIER SUMMARY] band_size: " + bandSize +
                "\n[CLASSIFIER SUMMARY] best_param: " + bestParamId;
    }

    @Override
    public double distance(final Sequence first, final Sequence second) {
        return distComputer.distance(first.data[0], second.data[0], this.g, this.bandSize);
    }

    @Override
    public double distance(final Sequence first, final Sequence second, final double cutOffValue) {
        return distComputer.distance(first.data[0], second.data[0], this.g, this.bandSize);
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

        final LazyAssessNNERP[] lazyAssessNNS = new LazyAssessNNERP[train.size()];
        for (int i = 0; i < train.size(); ++i) {
            lazyAssessNNS[i] = new LazyAssessNNERP(cache);
        }
        final ArrayList<LazyAssessNNERP> challengers = new ArrayList<>(train.size());

        for (int current = 1; current < train.size(); ++current) {
            final Sequence sCurrent = train.get(current);

            challengers.clear();
            for (int previous = 0; previous < current; ++previous) {
                final LazyAssessNNERP d = lazyAssessNNS[previous];
                d.set(train.get(previous), previous, sCurrent, current);
                challengers.add(d);
            }

            for (int paramId = nParams - 1; paramId > -1; --paramId) {
                setParamsFromParamId(paramId);
                final int band = distComputer.getWindowSize(train.length(), bandSize);
                final CandidateNN currPNN = candidateNNS[paramId][current];

                if (currPNN.isNN()) {
                    // --- --- WITH NN CASE --- ---
                    // We already have the NN for sure, but we still have to check if current is the new NN for previous
                    for (int previous = 0; previous < current; ++previous) {
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- Try to beat the previous best NN
                        final double toBeat = prevNN.distance;
                        final LazyAssessNNERP challenger = lazyAssessNNS[previous];
                        final LazyAssessNNERP.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.g, this.bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNNERP.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(r);
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

                    for (LazyAssessNNERP challenger : challengers) {
                        final int previous = challenger.indexQuery;
                        final CandidateNN prevNN = candidateNNS[paramId][previous];

                        // --- First we want to beat the current best candidate:
                        double toBeat = currPNN.distance;
                        LazyAssessNNERP.RefineReturnType rrt = challenger.tryToBeat(toBeat, this.g, this.bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNNERP.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(r);
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
                        rrt = challenger.tryToBeat(toBeat, this.g, this.bandSize);

                        // --- Check the result
                        if (rrt == LazyAssessNNERP.RefineReturnType.New_best) {
                            final int r = challenger.getMinWindowValidityForFullDistance();
                            final double d = challenger.getDistance(r);
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
                    final double prevG = g;
                    int w = ERP.getWindowSize(train.length(), bandSize);
                    int tmp = paramId;
                    while (tmp > 0 && paramId % 10 > 0 && prevG == g && w >= r) {
                        candidateNNS[tmp][current].set(index, r, d, CandidateNN.Status.NN);
                        classCounts[tmp][current] = classCounts[paramId][current].clone();
                        tmp--;
                        this.setParamsFromParamId(tmp);
                        w = ERP.getWindowSize(train.length(), bandSize);
                    }
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
        if (!this.gAndWindowsRefreshed) {
            double stdv = GenericTools.stdv_p(trainData);
            bandSizes = GenericTools.getInclusive10(0, 0.25);
            gValues = GenericTools.getInclusive10(0.2 * stdv, stdv);
            this.gAndWindowsRefreshed = true;
        }
        this.g = gValues[paramId / 10];
        this.bandSize = bandSizes[paramId % 10];
        this.window = ERP.getWindowSize(trainData.length(), bandSize);
    }

    @Override
    public String getParamInformationString() {
        return "g=" + this.g + ", bandSize=" + this.bandSize;
    }

    protected int getParamIdFromWindow(final int currentParamId, final int w, final int n) {
        double r = 1.0 * w / n;
        int i = currentParamId;
        while (i >= 0 && bandSizes[i % 10] != r)
            i--;

        return i;
    }
}
