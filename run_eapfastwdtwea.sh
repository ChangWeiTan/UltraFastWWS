#!/bin/bash

cpu=87
iter=0

javac -sourcepath src -cp lib/* -d bin src/experiments/TrainingTimeBenchmark.java

cd bin

java -cp ../lib/*: experiments.TrainingTimeBenchmark -classifier=EAPFastWDTWEA -cpu=${cpu} -iter=${iter} -verbose=1 -eval=false -problem="all"