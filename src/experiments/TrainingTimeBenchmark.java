package experiments;

import application.Application;
import classifiers.TimeSeriesClassifier;
import datasets.DatasetLoader;
import datasets.Sequences;
import datasets.TimeSeriesDatasets;
import multiThreading.BenchmarkTask;
import multiThreading.MultiThreadedTask;
import results.ClassificationResults;
import results.TrainingClassificationResults;
import utils.StrLong;

import java.util.*;
import java.util.concurrent.Callable;

import static application.Application.extractArguments;
import static utils.GenericTools.doTimeNs;
import static utils.GenericTools.println;

public class TrainingTimeBenchmark {
    static String moduleName = "TrainingTimeBenchmark";
    private static final String[] testArgs = new String[]{
            "-problem=Adiac",
//            "-classifier=UltraFastTWE2", // see classifiers in TimeSeriesClassifier.java
//            "-classifier=UltraFastERP3",
//            "-classifier=UltraFastLCSS2",
//            "-classifier=UltraFastLCSS3",
//            "-classifier=TWELOOCV",
//            "-classifier=UltraFastWWSearchV2",
            "-classifier=UltraFastMSM3",
            "-paramId=-1",
            "-cpu=1",
            "-verbose=1",
            "-iter=0",
            "-retrain=true",
            "-eval=false",
    };

    public static void main(String[] args) throws Exception {
        final long startTime = System.nanoTime();
        args = testArgs;
        extractArguments(args);

        if (Application.problem.equals(""))
            Application.problem = "Trace";

        Application.printSummary(moduleName);

        switch (Application.problem) {
            case "all":
                if (Application.numThreads == 1) {
                    StrLong[] datasetOps = TimeSeriesDatasets.allDatasetOperations;
                    Arrays.sort(datasetOps);
                    for (int i = datasetOps.length-1; i >= 0; i--) {
                        StrLong a = datasetOps[i];
                        singleRun(a.str);
                        Application.outputPath = null;
                    }
                    break;
                }
            case "small":
                if (Application.numThreads == 1) {
                    StrLong[] datasetOps = TimeSeriesDatasets.smallDatasetOperations;
                    Arrays.sort(datasetOps);
                    for (int i = datasetOps.length-1; i >= 0; i--) {
                        StrLong a = datasetOps[i];
                        singleRun(a.str);
                        Application.outputPath = null;
                    }
                    break;
                }
                String[] datasets;
                StrLong[] datasetOps;
                if (Application.problem.equals("small")) {
                    datasets = TimeSeriesDatasets.smallDatasets;
                    datasetOps = TimeSeriesDatasets.smallDatasetOperations;
                } else {
                    datasets = TimeSeriesDatasets.allDatasets;
                    datasetOps = TimeSeriesDatasets.allDatasetOperations;
                }
                Arrays.sort(datasetOps);
                long totalOp = 0;
                for (StrLong s : datasetOps) totalOp += s.value;

                println("[" + moduleName + "] Number of datasets: " + datasets.length);
                println("[" + moduleName + "] Total operations: " + totalOp);

                ArrayList<String> myList = new ArrayList<>();
                Collections.addAll(myList, datasets);
                Collections.shuffle(myList, new Random(42));
// Setup parallel training tasks
                int numThreads = Application.numThreads;
                if (numThreads <= 0) numThreads = Runtime.getRuntime().availableProcessors();
                numThreads = Math.min(numThreads, Runtime.getRuntime().availableProcessors());

                long operationPerThread = totalOp / numThreads;

                println("[" + moduleName + "] Number of threads: " + numThreads);
                println("[" + moduleName + "] Operations per thread: " + operationPerThread);

                final MultiThreadedTask parallelTasks = new MultiThreadedTask(numThreads);

                List<Callable<Integer>> tasks = new ArrayList<>();
                ArrayList<String>[] subset = new ArrayList[numThreads];
                for (int i = 0; i < numThreads; i++)
                    subset[i] = new ArrayList<>();
                int threadCount = 0;
                for (StrLong s : datasetOps) {
                    subset[threadCount].add(s.str);
                    threadCount++;
                    if (threadCount == numThreads) threadCount = 0;
                }
                for (int i = 0; i < numThreads; i++) {
                    String[] tmp = new String[subset[i].size()];
                    for (int j = 0; j < subset[i].size(); j++) {
                        tmp[j] = subset[i].get(subset[i].size() - j - 1);
                    }
                    tasks.add(new BenchmarkTask(tmp, i));
                }
                MultiThreadedTask.invokeParallelTasks(tasks, parallelTasks);
                parallelTasks.getExecutor().shutdown();
                break;
            case "test":
                quickTest();
                break;
            default:
                singleRun(Application.problem);
                break;
        }
        final long endTime = System.nanoTime();
        println("[" + moduleName + "] Total time taken " + doTimeNs(endTime - startTime));
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

        if (!Application.retrain && Application.isDatasetDone(Application.outputPath))
            return;

        DatasetLoader loader = new DatasetLoader();
        Sequences trainData = loader.readUCRTrain(problem, Application.datasetPath, Application.znorm).reorderClassLabels(null);
        trainData.summary();

        TimeSeriesClassifier classifier = Application.initTSC(trainData);
        println(classifier);

        TrainingClassificationResults trainingResults = classifier.fit(trainData);
        trainingResults.problem = problem;
        println("[" + moduleName + "]" + trainingResults);
        println(classifier);

        double totalTime = trainingResults.elapsedTimeNanoSeconds;
        if (Application.doEvaluation) {
            Sequences testData = loader.readUCRTest(problem, Application.datasetPath, Application.znorm).reorderClassLabels(trainData.getInitialClassLabels());
            testData.summary();
            ClassificationResults classificationResults = classifier.evaluate(testData);
            testData.summary();
            classificationResults.problem = problem;
            println("[" + moduleName + "]" + classificationResults);
            totalTime += classificationResults.elapsedTimeNanoSeconds;
            Application.saveResults(
                    Application.outputPath,
                    trainingResults,
                    classificationResults);
        } else {
            Application.saveResults(
                    Application.outputPath,
                    trainingResults);
        }
        println("Total time taken " + totalTime);
    }

    private static void quickTest() throws Exception {
        /**
         * A quick test for speed and making sure the paramIDs are correct
         */
        String[] problems = new String[]{
                "ArrowHead",
                "ECG200",
                "ShapeletSim",
                "MiddlePhalanxTW",
                "Chinatown",
                "BME",
                "Beef",
                "BeetleFly",
                "GunPointOldVersusYoung",
                "FaceFour",
                "GunPoint",
                "MedicalImages",
                "OSULeaf",
                "GunPointMaleVersusFemale",
                "SwedishLeaf",
                "Adiac",
                "CricketX",
                "CricketY",
                "CricketZ",
                "FiftyWords",
                "ChlorineConcentration",
                "Computers",
        };
        int[] bestParams = new int[]{
                0,
                0,
                3,
                3,
                0,
                4,
                0,
                7,
                4,
                2,
                0,
                20,
                7,
                0,
                2,
                3,
                10,
                17,
                5,
                6,
                0,
                12,
        };

        int passed = 0;
        for (int i = 0; i < problems.length; i++) {
            String problem = problems[i];
            int bestParam = bestParams[i];
            if (Application.outputPath == null) {
                if (Application.paramId > 0)
                    Application.outputPath = System.getProperty("user.dir") +
                            "/outputs/benchmark (test)/" +
                            Application.classifierName + "_" +
                            Application.paramId + "/" +
                            Application.iteration + "/" +
                            problem + "/";
                else
                    Application.outputPath = System.getProperty("user.dir") +
                            "/outputs/benchmark (test)/" +
                            Application.classifierName + "/" +
                            Application.iteration + "/" +
                            problem + "/";
            }
            System.out.print("[" + moduleName + "] Problem=" + problem);
            DatasetLoader loader = new DatasetLoader();
            Sequences trainData = loader.readUCRTrain(problem, Application.datasetPath, Application.znorm);

            println(", Length=" + trainData.length());

            TimeSeriesClassifier classifier = Application.initTSC(trainData);

            TrainingClassificationResults trainingResults = classifier.fit(trainData);
            trainingResults.problem = problem;

            if (bestParam == trainingResults.paramId) {
                println("[" + moduleName + "] " + problem + " passed, time: " + trainingResults.elapsedTimeMilliSeconds);
                passed++;
            } else {
                println("[" + moduleName + "] " + problem + " failed, " + bestParam + " vs " +
                        trainingResults.paramId + "(" + (Math.ceil(100.0 * trainingResults.paramId / trainData.length())) +
                        ")" + ", time: " + trainingResults.elapsedTimeMilliSeconds);
            }
        }
        println(passed + " out of " + problems.length + " passed!");
    }
}
