#include "testing.h"

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

void generate_classify(int num, double l_x1, double u_x1, double l_x2, double u_x2,  double noise, Eigen::MatrixXd& x1, Eigen::MatrixXd& x2, Eigen::MatrixXd& label){
	Eigen::MatrixXd rand_x1(1,num);
	Eigen::MatrixXd rand_x2(1,num);
	Eigen::MatrixXd rand_label(1,num);
	double temp_x1, temp_x2, temp_noise, mult;
	
	auto seed_val = hrclock::now().time_since_epoch().count(); 	
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution_x1(l_x1,u_x1);
	std::uniform_real_distribution<double> distribution_x2(l_x2,u_x2);
	std::uniform_real_distribution<double> distribution_noise(0,1);
	generator.seed(distribution_x1(generator)*seed_val); 
	for (int i=0; i<num; i++){
		temp_x1 = distribution_x1(generator);
		generator.seed(temp_x1 * seed_val); 
		temp_x2 = distribution_x2(generator);
		generator.seed(temp_x2 * seed_val); 
		temp_noise = distribution_noise(generator); 
		generator.seed(temp_noise * seed_val); 

		(temp_noise >= noise) ? mult = 1.0 : mult = -1.0; 
		(temp_x2 >= (pow(temp_x1,3) - 3*pow(temp_x1,2) + 2*temp_x1)) ? rand_label (0,i) = mult*1.0 : rand_label(0,i) = mult*-1.0; 
		
		rand_x1(0,i) = temp_x1;
		rand_x2(0,i) = temp_x2; 
	}
	x1 = rand_x1; 
	x2 = rand_x2; 
	label = rand_label;
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
	
	generate_classify(num_points, 0, 2.5, -1, 2, 0.1, x1, x2, y); 
	initialise(hclass, x1, x2, feat, weights_d); 
	return test_data(feat, G, y); 

}
