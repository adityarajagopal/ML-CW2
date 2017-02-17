#ifndef GENERATE_DATA_H
#define GENERATE_DATA_H

#include <eigen3/Eigen/Dense> 

void generate_data(int num, double lower, double upper, Eigen::MatrixXd& points); 
void generate_x1(int num, double lower, double upper, Eigen::MatrixXd& points);
double test_data(const Eigen::MatrixXd& X, const Eigen::MatrixXd& W, const Eigen::MatrixXd& y);
double test_hypothesis(int num_points, double hclass, const Eigen::MatrixXd& G);
double test_data(const Eigen::MatrixXd& X, const Eigen::MatrixXd& W, const Eigen::MatrixXd& y);

#endif
