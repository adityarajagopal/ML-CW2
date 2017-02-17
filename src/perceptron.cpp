#include "perceptron.h"
#include <chrono>
#include <random>
#include <iostream>

typedef std::chrono::high_resolution_clock hrclock;

void classify (const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2, double noise, Eigen::MatrixXd& label, Eigen::MatrixXi& colour, Eigen::MatrixXd& orig){
	Eigen::MatrixXd class_temp(1,x1.cols()); 
	Eigen::MatrixXd orig_line = x1.array().pow(3) - 3*x1.array().pow(2) + 2*x1.array(); 
	double test, mult;

	auto seed_val = hrclock::now().time_since_epoch().count(); 	
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0,1);
	generator.seed(distribution(generator) * seed_val);
	class_temp = x2.array() - orig_line.array(); 
	for (size_t i=0, size=x2.size(); i < size; i++){
		test = distribution(generator);	
		(test >= noise) ? mult = 1 : mult = -1; 
		
		if (*(class_temp.data()+i) >= 0){
			*(class_temp.data() + i) = mult*1; 
		}
		else{
			*(class_temp.data()+i) = mult*-1; 
		}
		generator.seed(test * seed_val);
	}
	
	label = class_temp; 
	colour = class_temp.cast<int>(); 
	orig = orig_line; 
}

void initialise (int q, const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2, Eigen::MatrixXd& feat, Eigen::MatrixXd& weights){
	Eigen::MatrixXd feat_temp(q+2, x1.cols()); 
	Eigen::MatrixXd w_temp(q+2,1); 
	
	w_temp.col(0).array() = 1; 
	feat_temp.row(0).array() = 1; 
	feat_temp.row(q+1) = x2; 
	
	for (int i=1; i<=q; i++){
		feat_temp.row(i) = x1.array().pow(i); 	
	}	
	feat = feat_temp; 
	weights = w_temp; 
}

void percept(const Eigen::MatrixXd& X, Eigen::MatrixXd& w, const Eigen::MatrixXd& y, int limit, double& error){
	double curr_error = 0; 
	double min_error = 2; 
	int iter = 0; 
	double total_points = y.cols(); 
	Eigen::MatrixXd::Index error_col; 
	Eigen::MatrixXd::Index error_row; 
	Eigen::MatrixXd h(1, y.cols()); 
	Eigen::MatrixXd e_vec(1, y.cols()); 
	Eigen::MatrixXd min_w(w.rows(), 1); 

	for (iter=0; iter<limit; iter++){
		h = w.transpose() * X; 		
		h = h.unaryExpr(std::ptr_fun(signum)); 
		e_vec = h.array() + y.array(); 
		curr_error = (e_vec.array() == 0).count() / total_points; 
			
		if (curr_error < min_error){
			min_error = curr_error;
			min_w = w; 
		}
		
		if (curr_error == 0){
			std::cerr << "here" << std::endl; 
			break;
		}
		else{
			e_vec.array().abs().minCoeff(&error_row, &error_col); 
			w = w + y(0,error_col) * X.col(error_col); 
		}
	}
	error = min_error; 
	w = min_w;
}

double signum (double x){
	if (x >= 0){
		return 1.0; 
	}
	else {
		return -1.0; 
	}
}

























