#include <iostream> 
#include <eigen3/Eigen/Dense>
#include "generate_data.h"

int main(){
	double a = 4;
	double b = 3;
	Eigen::MatrixXd x1; 
	Eigen::MatrixXd x2; 

	generate_data(10,0,2.5,x1);
	generate_data(10,-1,2,x2);

	std::cout << x1 << std::endl; 
	std::cout << x2 << std::endl; 

	return 0; 
}
