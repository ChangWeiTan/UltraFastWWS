package experiments;

import application.Application;
import datasets.DatasetLoader;
import datasets.Sequence;
import datasets.Sequences;
import datasets.TimeSeriesDatasets;
import distances.classic.MSM;
import distances.classic.TWE;
import distances.eap.EAPMSM;
import distances.eap.EAPTWE;

import static application.Application.extractArguments;
import static classifiers.classicNN.MSM1NN.msmParams;
import static classifiers.classicNN.TWE1NN.tweLamdaParams;
import static classifiers.classicNN.TWE1NN.tweNuParams;
import static utils.GenericTools.doTimeNs;

public class TestEAP {
    static String moduleName = "TrainingTimeBenchmark";
    private static final String[] testArgs = new String[]{
            "-problem=ECG5000",
            "-paramId=99",
            "-cpu=4",
            "-verbose=1",
            "-iter=0",
            "-eval=false",
    };

    public static void main(String[] args) throws Exception {
        args = testArgs;
        extractArguments(args);

        if (Application.problem.equals(""))
            Application.problem = "Trace";

        Application.printSummary(moduleName);

        switch (Application.problem) {
            case "all":
                for (String problem : TimeSeriesDatasets.allDatasets)
                    singleRun(problem);
                break;
            case "small":
                for (String problem : TimeSeriesDatasets.smallDatasets)
                    singleRun(problem);
                break;
            default:
                singleRun(Application.problem);
                break;
        }
    }

    /**
     * Single run of the experiments
     */
    private static void singleRun(String problem) throws Exception {
        if (Application.outputPath == null) {
            if (Application.paramId > 0)
                Application.outputPath = System.getProperty("user.dir") +
                        "/outputs/benchmark/" +
                        Application.classifierName + "_" +
                        Application.paramId + "/" +
                        Application.iteration + "/" +
                        problem + "/";
            else
                Application.outputPath = System.getProperty("user.dir") +
                        "/outputs/benchmark/" +
                        Application.classifierName + "/" +
                        Application.iteration + "/" +
                        problem + "/";
        }

        DatasetLoader loader = new DatasetLoader();
        Sequences trainData = loader.readUCRTrain(problem, Application.datasetPath, Application.znorm);
        trainData.summary();

        msmUBRun(trainData);
    }

    private static void msmRun(Sequences trainData) {
        MSM distComputer = new MSM();
        EAPMSM eapDistComputer = new EAPMSM();
        for (int p = 0; p < 10; p++) {
            double c = msmParams[p];

            // normal distance with EA
            System.out.print("Running normal distance without EA, C=" + c);
            long startTime = System.nanoTime();
            for (int i = 0; i < trainData.size(); i++) {
                double bestSoFar = Double.POSITIVE_INFINITY;
                for (int j = 0; j < trainData.size(); j++) {
                    if (i == j) continue;
                    double d = distComputer.distance(trainData.get(i).data[0], trainData.get(j).data[0], c, Double.POSITIVE_INFINITY);
                    if (d < bestSoFar) {
                        bestSoFar = d;
                    }
                }
            }
            long endTime = System.nanoTime();
            System.out.println(", took " + doTimeNs(endTime - startTime));

            // eap distance with EA
            System.out.print("Running normal distance with EA, C=" + c);
            startTime = System.nanoTime();
            for (int i = 0; i < trainData.size(); i++) {
                double bestSoFar = Double.POSITIVE_INFINITY;
                for (int j = 0; j < trainData.size(); j++) {
                    if (i == j) continue;
                    double d = distComputer.distance(trainData.get(i).data[0], trainData.get(j).data[0], c, bestSoFar);
                    if (d < bestSoFar) {
                        bestSoFar = d;
                    }
                }
            }
            endTime = System.nanoTime();
            System.out.println(", took " + doTimeNs(endTime - startTime));

            // eap distance without EA
            System.out.print("Running EAP distance without EA, C=" + c);
            startTime = System.nanoTime();
            for (int i = 0; i < trainData.size(); i++) {
                double bestSoFar = Double.POSITIVE_INFINITY;
                for (int j = 0; j < trainData.size(); j++) {
                    if (i == j) continue;
                    double d = eapDistComputer.distance(trainData.get(i).data[0], trainData.get(j).data[0], c, Double.POSITIVE_INFINITY);
                    if (d < bestSoFar) {
                        bestSoFar = d;
                    }
                }
            }
            endTime = System.nanoTime();
            System.out.println(", took " + doTimeNs(endTime - startTime));

            // eap distance with EA
            System.out.print("Running EAP distance with EA, C=" + c);
            startTime = System.nanoTime();
            for (int i = 0; i < trainData.size(); i++) {
                double bestSoFar = Double.POSITIVE_INFINITY;
                for (int j = 0; j < trainData.size(); j++) {
                    if (i == j) continue;
                    double d = eapDistComputer.distance(trainData.get(i).data[0], trainData.get(j).data[0], c, bestSoFar);
                    if (d < bestSoFar) {
                        bestSoFar = d;
                    }
                }
            }
            endTime = System.nanoTime();
            System.out.println(", took " + doTimeNs(endTime - startTime));
        }
    }

    private static void msmUBRun(Sequences trainData) {
        EAPMSM eapDistComputer = new EAPMSM();
        double c = msmParams[99];

        System.out.println("Running distance with, c=" + c);
        long startTime = System.nanoTime();
        long t;
        double t1=0, t2=0;
        for (int i = 0; i < trainData.size(); i++) {
            for (int j = 0; j < trainData.size(); j++) {
                if (i == j) continue;
                t = System.nanoTime();
                double d = eapDistComputer.l1(trainData.get(i).data[0], trainData.get(j).data[0]);
                t1 += (System.nanoTime() - t) / 1e9;

                t = System.nanoTime();
                double d2 = eapDistComputer.distance(trainData.get(i).data[0], trainData.get(j).data[0], c, Double.POSITIVE_INFINITY);
                t2 += (System.nanoTime() - t) / 1e9;

                if (d != d2) {
                    System.out.println(i + "," + j + ", d=" + d + ",Eap=" + d2);
                }
            }
        }
        long endTime = System.nanoTime();
        System.out.println(", took " + doTimeNs(endTime - startTime));

        System.out.println("t1 " + t1);
        System.out.println("t2 " + t2);
    }

    private static void tweRun(Sequences trainData) {
        TWE distComputer = new TWE();
        EAPTWE eapDistComputer = new EAPTWE();
        for (int p = 0; p < 1; p++) {
            double nu = tweNuParams[p / 10];
            double lambda = tweLamdaParams[p % 10];

            System.out.println("Running distance with, Nu=" + nu + ", Lamda=" + lambda);
            long startTime = System.nanoTime();
            for (int i = 0; i < trainData.size(); i++) {
                for (int j = 17; j < trainData.size(); j++) {
                    if (i == j) continue;
                    double d = distComputer.distance(trainData.get(i).data[0], trainData.get(j).data[0], nu, lambda, Double.POSITIVE_INFINITY);
                    double d2 = eapDistComputer.distance(trainData.get(i).data[0], trainData.get(j).data[0], nu, lambda, Double.POSITIVE_INFINITY);

                    if (d != d2) {
                        System.out.println(i + "," + j + ", d=" + d + ",Eap=" + d2);
                    }
                    break;
                }
                break;
            }
            long endTime = System.nanoTime();
            System.out.println(", took " + doTimeNs(endTime - startTime));
        }
    }
}
