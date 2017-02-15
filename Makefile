#!/bin/bash

bin/data : obj/main.o obj/generate_data.o
	@g++ -o bin/data obj/main.o obj/generate_data.o

obj/main.o : src/main.cpp src/generate_data.h
	@g++ -c src/main.cpp -o obj/main.o

obj/generate_data.o : src/generate_data.cpp src/generate_data.h
	@g++ -c src/generate_data.cpp -o obj/generate_data.o

run : bin/data
	@./bin/data

clean : 
	@rm bin/*
	@rm obj/*
	

