#!/bin/bash

bin/data : obj/main.o obj/testing.o obj/perceptron.o
	@g++ -o bin/data obj/main.o obj/testing.o obj/perceptron.o

obj/main.o : src/main.cpp src/testing.h
	@g++ -std=c++11 -c src/main.cpp -o obj/main.o

obj/testing.o : src/testing.cpp src/testing.h
	@g++ -std=c++11 -c src/testing.cpp -o obj/testing.o

obj/perceptron.o : src/perceptron.cpp src/perceptron.h
	@g++ -std=c++11 -c src/perceptron.cpp -o obj/perceptron.o

run : bin/data
	#@./bin/data 1> "files/graphs.json" 2> "files/log.txt"
	@./bin/data 2> "files/log.txt"
	#@python src/print.py

clean : 
	@rm bin/*
	@rm obj/*
	

