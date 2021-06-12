package experiments;

import application.Application;
import datasets.TimeSeriesDatasets;
import multiThreading.MultiThreadedTask;
import multiThreading.ScalabilitySizeTask;
import utils.StrDouble;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import static application.Application.extractArguments;
import static utils.GenericTools.doTimeNs;
import static utils.GenericTools.println;

public class ScalabilityTrainSize {
    static String moduleName = "FASTEREE";
    private static final String[] testArgs = new String[]{
            "-problem=SITS1M_fold1",
//            "-classifier=EAPDTW_1NN-FastCV_EA_NoLb_NNOrder-LbKeoghV9",
            "-classifier=EAPDTW_1NN-FastCV_EA_NoLb-LbKeoghV1-NNOrderV2",
            "-paramId=99",
            "-cpu=2",
            "-verbose=2",
            "-iter=0",
            "-eval=0",
            "-trainSize=0.01",
    };

    public static void main(String[] args) throws Exception {
        final long startTime = System.nanoTime();
//        args = testArgs;
        extractArguments(args);

        if (Application.datasetPath == null) {
            String osName = System.getProperty("os.name");
            String username = System.getProperty("user.name");
            if (osName.contains("Window")) {
                Application.datasetPath = "C:/Users/" + username + "/workspace/Dataset/UCRArchive_2018/";
            } else {
                Application.datasetPath = "/home/" + username + "/workspace/Dataset/UCRArchive_2018/";
            }
        }

        Application.printSummary(moduleName);
        String[] datasets;
        if (Application.problem.equals("all")) {
            datasets = TimeSeriesDatasets.largeDatasets;
            println("[" + moduleName + "] Number of datasets: " + datasets.length);
        } else {
            datasets = new String[]{Application.problem};
        }

        // Setup parallel training tasks
        int numThreads = Application.numThreads;
        if (numThreads <= 0) numThreads = Runtime.getRuntime().availableProcessors();
        numThreads = Math.min(numThreads, Runtime.getRuntime().availableProcessors());

        int operationsPerThread = 1;
        if (Application.scalabilityTrainRatio == 0) {
            operationsPerThread = 10 * datasets.length / numThreads;
            if (operationsPerThread < 1) numThreads = 10 * datasets.length;
        } else {
            numThreads = datasets.length;
        }

        println("[" + moduleName + "] Number of threads: " + numThreads);
        println("[" + moduleName + "] Datasets per thread: " + operationsPerThread);

        final MultiThreadedTask parallelTasks = new MultiThreadedTask(numThreads);
        List<Callable<Integer>> tasks = new ArrayList<>();
        ArrayList<StrDouble>[] subset = new ArrayList[numThreads];
        for (int i = 0; i < numThreads; i++)
            subset[i] = new ArrayList<>();

        int threadCount = 0;

        for (String s : datasets) {
            if (Application.scalabilityTrainRatio == 0) {
                if (Application.problem.contains("SITS1M"))
                    for (int i = 1; i <= 10; i++) {
                        subset[threadCount].add(new StrDouble(s, 1.0 * i / 100));
                        threadCount++;
                        if (threadCount == numThreads) threadCount = 0;
                    }
                else
                    for (int i = 1; i <= 10; i++) {
                        subset[threadCount].add(new StrDouble(s, 1.0 * i / 10));
                        threadCount++;
                        if (threadCount == numThreads) threadCount = 0;
                    }
            } else {
                subset[threadCount].add(new StrDouble(s, Application.scalabilityTrainRatio));
                threadCount++;
            }
        }
        for (int i = 0; i < numThreads; i++) {
            String[] tmp = new String[subset[i].size()];
            double[] tmp2 = new double[subset[i].size()];
            for (int j = 0; j < subset[i].size(); j++) {
                tmp[j] = subset[i].get(j).str;
                tmp2[j] = subset[i].get(j).value;
            }
            tasks.add(new ScalabilitySizeTask(tmp, i, tmp2));
        }
        MultiThreadedTask.invokeParallelTasks(tasks, parallelTasks);
        parallelTasks.getExecutor().shutdown();

        final long endTime = System.nanoTime();
        println("[" + moduleName + "] Total time taken " + doTimeNs(endTime - startTime));
    }
}
