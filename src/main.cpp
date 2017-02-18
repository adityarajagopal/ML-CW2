#include <iostream> 
#include <vector>
#include <chrono>
#include <cmath>

#include "Eigen/Dense"
#include "testing.h"
#include "perceptron.h"
#include "json.hpp"

using json = nlohmann::json;
typedef std::chrono::high_resolution_clock hrclock;

void generate_graph(std::vector<double>& v_x, std::vector<double>& v_y, std::string title, std::string x_lab, std::string y_lab, std::string legend, json& JSON, std::vector<double>& colour, std::vector<double>& original, std::vector<double>& learned); 

double complexity(double v_c, double n, double delta, double w_i);

int main(){
	Eigen::MatrixXd x1; 
	Eigen::MatrixXd x2; 
	Eigen::MatrixXd y; 
	Eigen::MatrixXd original; 
	Eigen::MatrixXd feat; 
	Eigen::MatrixXd weights; 
	Eigen::MatrixXd learned; 
	Eigen::MatrixXd g_srm; 
	Eigen::MatrixXi colour; 
	double error_percent, noise, avg_error, delta, h_w, avg_min;
	double srm, erm, srm_min, iterations, omega;
	int q, num_points, selected_class;  
	std::vector<double> v1, v2, vo, vc, vl, r_h, omega_erm, omega_srm; 
	json graph; 
	
	noise = 0.1; 
	num_points = 100000;
	iterations = 10;
	delta = 0.5;
	h_w = 0.2; 
	srm_min = 2; 
	
	for (q = 0; q < 5; q++){
		std::cout << "q: " << q << std::endl;
		Eigen::MatrixXd curr_g(q+2,(int)iterations); 
		avg_error = 0;

		auto start = hrclock::now();
		for (int i=0; i<iterations; i++){
			std::cout << "iter num: " << i << std::endl; 
			//generate_data(num_points,0,2.5,x1);
			//generate_data(num_points,-1,2,x2);
			//classify(x1, x2, noise, y, colour, original);
			generate_classify(num_points, 0, 2.5, -1, 2, noise, x1, x2, y); 
			initialise(q, x1, x2, feat, weights); 
			percept(feat, weights, y, 10000, error_percent); 
			avg_error += error_percent; 
			curr_g.col(i) = weights;
		}
		auto end = hrclock::now();
		
		//std::cout << x1 << std::endl; 
		//std::cout << x2 << std::endl; 
		//std::cout << y << std::endl; 
		avg_error = avg_error/iterations; 
		omega = complexity(q+2, num_points, delta, h_w);
		
		srm = avg_error + 0.1*omega;
		std::cout << "comp: " << omega << std::endl;
		std::cout << "srm: " << omega << std::endl;
		
		if (srm <= srm_min){
			std::cout << "srm_minimum_q: " << q << std::endl; 
			selected_class = q;
			srm_min = srm; 
			g_srm = curr_g; 
		}
		
		std::cout << "minimum_srm: " << srm_min << std::endl;
		std::cout << "minimum_rn: " << avg_error<< std::endl; 
		std::cout << "time for iteration: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << std::endl; 
		std::cout << curr_g << std::endl;  
	}
	
	int test_num_points = 1000000;
	double te = test_hypothesis(test_num_points, selected_class, g_srm); 	

	std::cout << te << std::endl; 
		
	return 0;
}

void generate_graph(std::vector<double>& v_x, std::vector<double>& v_y, std::string title, std::string x_lab, std::string y_lab, std::string legend, json& JSON, std::vector<double>& colour, std::vector<double>& original, std::vector<double>& learned){
	JSON = 
	{
		{"axes",
			{
				{"x",v_x},
				{"y",v_y}
			}
		}, 
		{"labels",
			{
				{"title",title}, 
				{"xlabel",x_lab}, 
				{"ylabel",y_lab},
				{"legend",legend},
				{"colour",colour},
				{"original",original},
				{"learned", learned}
			}
		}
	};
}

double complexity(double v_c, double n, double delta, double w_i){
	double a = ((8*v_c)/n)*log((2*n)+1); 
	double b = (8/n)*log((8/(delta*w_i))); 
	double c = (1/(2*n))*log((4/(w_i*delta))); 
	return (sqrt(a + b) + sqrt(c));
}
