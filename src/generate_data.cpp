#include "generate_data.h"

void generate_data(int num, double lower, double upper,  Eigen::MatrixXd& points){
	double scale = (upper - lower)/2; 
	double shift = lower + scale; 
	Eigen::MatrixXd offset = Eigen::MatrixXd::Constant(1,num,shift);
	std::srand((unsigned int) time(0));
	points = (scale * Eigen::MatrixXd::Random(1,num)) + offset; 	
}
