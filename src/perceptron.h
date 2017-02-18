#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "Eigen/Dense"
#include <chrono>
#include <random>
#include <iostream>

typedef std::chrono::high_resolution_clock hrclock;

//Inputs : x1, x2
//Outputs :label, colour
void classify (const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2, double noise, Eigen::MatrixXd& label, Eigen::MatrixXi& colour, Eigen::MatrixXd& orig);

void initialise (int q, const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2, Eigen::MatrixXd& feat, Eigen::MatrixXd& weights);

void percept(const Eigen::MatrixXd& X, Eigen::MatrixXd& w, const Eigen::MatrixXd& y, int limit, double& error);

double signum(double x); 

#endif
