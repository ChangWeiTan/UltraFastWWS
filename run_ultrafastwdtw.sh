#!/bin/bash

cpu=8
iter=0

javac -sourcepath src -cp lib/* -d bin src/experiments/TrainingTimeBenchmark.java

cd bin

java -cp ../lib/*: experiments.TrainingTimeBenchmark -classifier=UltraFastWDTW -cpu=${cpu} -iter=${iter} -verbose=0 -eval=false -problem="all"