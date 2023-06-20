#!/bin/bash

# all.sh 프로세스 종료
pid=$(ps -a | awk '/all.sh/ {print $1}')

#kill -9 $pid;

pids=$(ps -a | awk '/python/ {print $1}')

kill -9 $pids;

# 서버 종료
pid2=$(ps aux | grep uvicorn | grep main:app | awk '{print $2}')

kill -9 $pid2

pid1=$(sudo netstat -tulnp | awk '$4 == "0.0.0.0:8010" {split($7, a, "/"); print a[1]}')


kill -9 $pid1

uvicorn main:app --reload --host=0.0.0.0 --port=8010
