#!/bin/sh
g++ -lm -pthread -g -Ofast -funroll-loops -ffast-math -Wall -Wno-unused-result line.cpp -o line -lgsl -lgslcblas
