#include "testing.h"
#include "perceptron.h"
#include <random>
#include <chrono>
#include <iostream>

typedef std::chrono::high_resolution_clock hrclock;

void generate_data(int num, double lower, double upper, Eigen::MatrixXd& points){
	double scale = (upper - lower)/2; 
	double shift = lower + scale; 
	Eigen::MatrixXd offset = Eigen::MatrixXd::Constant(1,num,shift);
	Eigen::MatrixXd rand(1,num);
	
	auto seed_val = hrclock::now().time_since_epoch().count(); 	
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(lower,upper);
	generator.seed(distribution(generator)*seed_val); 
	for (int i=0; i<num; i++){
		rand(0,i) = distribution(generator); 
		generator.seed(rand(0,i)* seed_val); 
	}
	points = rand; 
}

double test_data(const Eigen::MatrixXd& X, const Eigen::MatrixXd& W, const Eigen::MatrixXd& y){
	Eigen::MatrixXd H = W.transpose() * X; 
	H = H.unaryExpr(std::ptr_fun(signum)); 
	Eigen::MatrixXd E(H.rows(), 1);
	double total_points = X.cols();  
	for (int i=0; i<H.rows(); i++){
		E(i,0) = ((H.row(i).array() + y.array())==0).count() / total_points; 
	}	
	return E.col(0).sum()/E.rows();
}

double test_hypothesis(int num_points, double hclass, const Eigen::MatrixXd& G){
	Eigen::MatrixXd x1; 
	Eigen::MatrixXd x2; 
	Eigen::MatrixXd y; 
	Eigen::MatrixXd original_d; 
	Eigen::MatrixXd feat; 
	Eigen::MatrixXd weights_d; 
	Eigen::MatrixXd learned; 
	Eigen::MatrixXi colour_d; 
	
	generate_data(num_points, 0, 2.5, x1); 
	generate_data(num_points, -1, 2, x2); 
	classify(x1, x2, 0.1, y, colour_d, original_d); 
	initialise(hclass, x1, x2, feat, weights_d); 
	return test_data(feat, G, y); 

}
