#ifndef GENERATE_DATA_H
#define GENERATE_DATA_H

#include "Eigen/Dense"
#include "perceptron.h"
#include <random>
#include <chrono>
#include <cmath>

typedef std::chrono::high_resolution_clock hrclock;

void generate_data(int num, double lower, double upper, Eigen::MatrixXd& points); 

void generate_classify(int num, double l_x1, double u_x1, double l_x2, double u_x2,  double noise, Eigen::MatrixXd& x1, Eigen::MatrixXd& x2, Eigen::MatrixXd& label);

double test_data(const Eigen::MatrixXd& X, const Eigen::MatrixXd& W, const Eigen::MatrixXd& y);

double test_hypothesis(int num_points, double hclass, const Eigen::MatrixXd& G);

double test_data(const Eigen::MatrixXd& X, const Eigen::MatrixXd& W, const Eigen::MatrixXd& y);

#endif
