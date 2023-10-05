#include <casadi/casadi.hpp>
#include <iostream>
#include <raisim/RaisimServer.hpp>
#include <ros/ros.h>

using std::cout;
using std::endl;

using namespace casadi;

double m = 3;
double g = 9.81;

double dT = 0.01;
int horizon = 3;
double u_max = 1.0;
double u_min = -u_max;

Function rock_mpc_ipopt()
{
  SX z = SX::sym("z");
  SX dz = SX::sym("dz");
  SX x = SX::vertcat({ z, dz });
  uint16_t x_num = 2;

  SX u = SX::sym("u");
  uint16_t u_num = 1;

  SX model = SX::vertcat({ dz + dT * 0.5 * (u / m - g), u / m - g });

  // Function f = Function("f", { x, u }, { model });
  Function f("f", { x, u }, { model });

  SX U = SX::sym("U", u_num, horizon);
  SX P = SX::sym("P", x_num * 2);
  SX X = SX::sym("X", x_num, horizon + 1);

  // Slice means START FROM, HOW MANY. If Slice(0,0) - empty result
  // Slice means START FROM, TO-1. If Slice(0,0) - empty result
  X(Slice(0, x_num), 0) = P(Slice(0, x_num));

  SX x_act = SX::sym("x_act", x_num, 1);
  SX x_next = SX::sym("x_next", x_num, 1);
  SX u_act = SX::sym("u_act", u_num, 1);

  std::vector<SX> input;
  input.push_back(x_act);
  input.push_back(u_act);

  for (uint32_t k = 0; k < horizon; k++)
  {
    x_act = X(Slice(0, x_num), k);
    u_act = U(Slice(0, u_num), k);

    input.clear();
    input.push_back(x_act);
    input.push_back(u_act);
    std::vector<SX> f_value = f(input);

    x_next = x_act + dT * f_value.at(0);

    X(Slice(0, x_num), k + 1) = x_next;
  }

  SX obj = 0;
  SX G = 0;

  SX Q = SX::zeros(x_num, x_num);
  Q(0, 0) = 1e3;
  Q(1, 1) = 1e1;
  SX R = SX::zeros(u_num, u_num);
  R(0, 0) = 1e-6;

  for (uint16_t k = 0; k < horizon; k++)
  {
    x_act = X(Slice(0, x_num), k);
    u_act = U(Slice(0, u_num), k);
    obj = obj + mtimes((x_act - P(Slice(x_num, x_num + x_num))).T(), mtimes(Q, x_act - P(Slice(x_num, x_num + x_num)))) + mtimes(u_act.T(), mtimes(R, u_act));
  }

  SX OPT_variables = SX::reshape(U, u_num * horizon, 1);

  // SXDict nlp_prob = { { "f", obj }, { "x", OPT_variables }, { "g", G }, { "p", P } };
  SXDict nlp_prob = { { "f", obj }, { "x", OPT_variables }, { "p", P } };

  Dict opts;
  Dict ipopt_opts;
  // opts["print_time"] = 1;
  opts["print_time"] = 0;
  ipopt_opts["max_iter"] = 1000;
  // ipopt_opts["print_level"] = 3;
  ipopt_opts["print_level"] = 0;
  ipopt_opts["acceptable_tol"] = 1e-8;
  ipopt_opts["acceptable_obj_change_tol"] = 1e-6;
  opts["ipopt"] = ipopt_opts;

  Function solver = nlpsol("solver", "ipopt", nlp_prob, opts);

  return solver;
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "rock_example_node");
  ros::NodeHandle n;
  ros::Rate rate(100);

  ROS_INFO("Init");

  raisim::World::setActivationKey("~/.raisim/activation.raisim");

  raisim::World world;
  raisim::RaisimServer server(&world);
  raisim::ArticulatedSystem* robot;

  world.setTimeStep(0.01);
  world.addGround();

  std::string urdf_path = "/home/splitmind/splitm_quadruped_ws/src/casadi_ipopt_examples/description/rock_model.urdf";
  robot = world.addArticulatedSystem(urdf_path);

  cout << "Robot DOF: " << robot->getDOF() << endl;

  Eigen::VectorXd spawn_pose;
  spawn_pose.resize(robot->getDOF() + 1);
  spawn_pose.setZero();

  Eigen::VectorXd spawn_velocity;
  spawn_velocity.resize(robot->getDOF());
  spawn_velocity.setZero();

  robot->setGeneralizedCoordinate(spawn_pose);
  robot->setGeneralizedVelocity(spawn_velocity);
  robot->setGeneralizedForce(Eigen::VectorXd::Zero(robot->getDOF()));
  robot->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
  robot->setName("rock");

  server.launchServer();
  server.focusOn(robot);

  Eigen::VectorXd pos_act;
  Eigen::VectorXd vel_act;
  Eigen::VectorXd u_des;
  u_des.resize(1);
  u_des(0) = 0;

  uint16_t x_num = 2;
  uint16_t u_num = 1;

  SX x_init = SX::sym("x_init", x_num, 1);
  SX x_des = SX::sym("x_des", x_num, 1);
  x_init(0) = 0;
  x_init(1) = 0;
  x_des(0) = 1;
  x_des(1) = 0;

  cout << "P: " << SX::vertcat({ x_init, x_des }) << endl;

  DMDict arg;
  arg["x0"] = SX::vertcat({ 0, 0, 0 });
  arg["lbg"] = -inf;
  arg["ubg"] = inf;
  arg["lbx"] = -50;
  arg["ubx"] = 50;
  arg["p"] = SX::vertcat({ x_init, x_des });

  Function solver = rock_mpc_ipopt();

  DMDict res = solver(arg);
  SX U0 = SX::zeros(u_num, horizon);
  SX U_previous = SX::zeros(u_num, horizon);
  // SX U0 = SX::sym("U0", u_num, horizon);
  // SX U_previous = SX::sym("U_previous", u_num, horizon);

  cout << "U pred: " << U0 << endl;
  // cout << "U previous: " << U_previous << endl;

  // U0(Slice(0, u_num), Slice(0, horizon - 1)) = U_previous(Slice(0, u_num), Slice(1, horizon));
  // U0(Slice(0, u_num), horizon - 1) = U_previous(Slice(0, u_num), horizon - 1);

  // cout << "U pred: " << U0 << endl;
  // cout << "U previous: " << U_previous << endl;

  //  Print solution
  // cout << "Optimal cost:                     " << double(res.at("f")) << endl;
  // cout << "Primal solution:                  " << std::vector<double>(res.at("x")) << endl;
  // cout << "Dual solution (simple bounds):    " << std::vector<double>(res.at("lam_x")) << endl;
  // cout << "Dual solution (nonlinear bounds): " << std::vector<double>(res.at("lam_g")) << endl;

  ROS_INFO("Enter loop");

  while (ros::ok())
  {
    server.integrateWorldThreadSafe();

    robot->getState(pos_act, vel_act);

    x_init(0) = pos_act(0);
    x_init(1) = vel_act(0);
    x_des(0) = -1;
    x_des(1) = 0;

    // arg["x0"] = (std::vector<double>(res.at("x"))).at(1);
    arg["x0"] = U0;
    arg["p"] = SX::vertcat({ x_init, x_des });
    cout << "p: " << SX::vertcat({ x_init, x_des }) << endl;

    res = solver(arg);

    U0 = res.at("x");
    cout << "Primal solution: " << std::vector<double>(U0) << endl;

    u_des(0) = std::vector<double>(U0).at(0);
    robot->setGeneralizedForce(u_des);

    // U0(Slice(0, u_num), Slice(0, horizon - 1)) = U0(Slice(0, u_num), Slice(1, horizon-1));
    U0(Slice(0, horizon - 1)) = U0(Slice(1, horizon));
    cout << "U0 shifted: " << U0 << endl;
    // U0(Slice(0, u_num), horizon - 1) = U0(Slice(0, u_num), horizon - 1);

    ros::spinOnce();
    rate.sleep();
  }

  server.killServer();

  return 0;
}
