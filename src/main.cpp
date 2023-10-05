#include <casadi/casadi.hpp>
#include <iostream>

using namespace casadi;

using std::cout;
using std::endl;

void test1()
{
  // Declare variables
  SX x = SX::sym("x");
  SX y = SX::sym("y");
  SX z = SX::sym("z");

  // Formulate the NLP
  SX f = pow(x, 2) + 100 * pow(z, 2);
  SX g = z + pow(1 - x, 2) - y;
  SXDict nlp = { { "x", SX::vertcat({ x, y, z }) }, { "f", f }, { "g", g } };

  // Set options
  Dict opts;
  // opts["expand"] = true;
  // opts["max_iter"] = 10;
  //  opts["verbose"] = true;
  //  opts["linear_solver"] = "ma57";
  //  opts["hessian_approximation"] = "limited-memory";
  //  opts["derivative_test"] = "second-order";

  // Create an NLP solver
  Function solver = nlpsol("solver", "ipopt", nlp, opts);

  // Solve the Rosenbrock problem
  DMDict arg;
  arg["x0"] = std::vector<double>{ 2.5, 3.0, 0.75 };
  arg["lbg"] = arg["ubg"] = 0;
  DMDict res = solver(arg);

  //  Print solution
  cout << "Optimal cost:                     " << double(res.at("f")) << endl;
  cout << "Primal solution:                  " << std::vector<double>(res.at("x")) << endl;
  cout << "Dual solution (simple bounds):    " << std::vector<double>(res.at("lam_x")) << endl;
  cout << "Dual solution (nonlinear bounds): " << std::vector<double>(res.at("lam_g")) << endl;
}

int main(int, char**)
{
  cout << "Hello, world" << endl;

  // test1();

  // Declare variables
  SX x = SX::sym("w");

  // Formulate the NLP
  SX f = pow(x, 2) - 6 * x + 13;
  SX g = 0;
  SX P;
  SXDict nlp_prob = { { "x", SX::vertcat({ x }) }, { "f", f }, { "g", g }, { "p", P } };

  Dict opts;
  Dict ipopt_opts;
  opts["print_time"] = 1;
  ipopt_opts["max_iter"] = 1000;
  ipopt_opts["print_level"] = 3;
  ipopt_opts["acceptable_tol"] = 1e-8;
  ipopt_opts["acceptable_obj_change_tol"] = 1e-6;
  opts["ipopt"] = ipopt_opts;

  // Create an NLP solver
  Function solver = nlpsol("solver", "ipopt", nlp_prob, opts);

  // Solve the Rosenbrock problem
  DMDict arg;
  arg["x0"] = -5;
  arg["lbg"] = -inf;
  arg["ubg"] = inf;
  arg["lbx"] = -inf;
  arg["ubx"] = inf;

  cout << "Before solving" << endl;
  DMDict res = solver(arg);

  //  Print solution
  cout << "Optimal cost:                     " << double(res.at("f")) << endl;
  cout << "Primal solution:                  " << std::vector<double>(res.at("x")) << endl;
  cout << "Dual solution (simple bounds):    " << std::vector<double>(res.at("lam_x")) << endl;
  cout << "Dual solution (nonlinear bounds): " << std::vector<double>(res.at("lam_g")) << endl;

  return 0;
}
