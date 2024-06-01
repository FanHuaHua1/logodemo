#! /bin/bash
export HADOOP_USER_NAME=caimeng
function join { local IFS="$1"; shift; echo "$*"; }

appName=$(join - $*)

echo submit $appName

queue=$1

shift

root=$PWD

# jars=$(join , `ls $PWD/targets/*jar`)
algojar=./rsp-algos-1.0-SNAPSHOT-jar-with-dependencies.jar

submitcmd="$SPARK_HOME/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--class com.szubd.rspalgos.App \
--name $appName \
$algojar $*"

if [ ! -d "$root/logs" ]
then
    mkdir $root/logs
fi

echo `date "+%Y-%m-%dT%T"` $submitcmd >> logs/submit-commands-history.log

$SPARK_HOME/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--class com.szubd.rspalgos.App \
--queue $queue \
--name $appName \
$algojar $*
