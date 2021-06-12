package experiments;

import application.Application;
import classifiers.TimeSeriesClassifier;
import datasets.DatasetLoader;
import datasets.Sequences;
import datasets.TimeSeriesDatasets;
import results.ClassificationResults;
import results.TrainingClassificationResults;

import static application.Application.extractArguments;
import static utils.GenericTools.doTimeNs;
import static utils.GenericTools.println;

public class TrainingTimeBenchmark {
    static String moduleName = "TrainingTimeBenchmark";
    private static final String[] testArgs = new String[]{
            "-problem=test",
            "-classifier=UltraFastWWSearch", // see classifiers in TimeSeriesClassifier.java
            "-paramId=99",
            "-cpu=-1",
            "-verbose=1",
            "-iter=0",
            "-eval=0",
    };

    public static void main(String[] args) throws Exception {
        final long startTime = System.nanoTime();
//        args = testArgs;
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

        DatasetLoader loader = new DatasetLoader();
        Sequences trainData = loader.readUCRTrain(problem, Application.datasetPath, Application.znorm);
        trainData.summary();

        TimeSeriesClassifier classifier = Application.initTSC(trainData);
        println(classifier);

        TrainingClassificationResults trainingResults = classifier.fit(trainData);
        trainingResults.problem = problem;
        println("[" + moduleName + "]" + trainingResults);

        double totalTime = trainingResults.elapsedTimeNanoSeconds;
        if (Application.doEvaluation) {
            Sequences testData = loader.readUCRTest(problem, Application.datasetPath, Application.znorm);
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
