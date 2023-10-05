#include <casadi/casadi.hpp>
#include <iostream>
#include <raisim/RaisimServer.hpp>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

using std::cout;
using std::endl;

#define LOOP_RATE 50
#define SIM_DT 0.02

using namespace casadi;

double m0 = 3;
double m1 = 1;
double m2 = 0;
double L = 0.25;
double r = 0.01;
double g = 9.80665;

double dT = 0.02;
int horizon = 100;
double u_max = 15.0;
double u_min = -u_max;

uint16_t x_num = 4;
uint16_t u_num = 1;

uint16_t num_pub_pred = 4;

std::vector<std::string> tf_prefix_names;
std::vector<SX> predicted_states;

ros::Publisher pub_predicted_state[4];

void pubPredictedState();

void initPredictionPub(ros::NodeHandle& _n)
{
  tf_prefix_names.resize(num_pub_pred);
  predicted_states.resize(num_pub_pred);

  std::string tf_base = "x_";

  for (uint16_t i = 0; i < num_pub_pred; i++)
  {
    tf_prefix_names.at(i) = tf_base + std::to_string(i);
    pub_predicted_state[i] = _n.advertise<sensor_msgs::JointState>(tf_prefix_names.at(i) + "/joint_state", 1);
  }
}

Function cartPendCasadiIpopt()
{
  double Iz = 1.0 / 3.0 * m1 * L * L;
  Iz = Iz + m2 * L * L;

  SX x = SX::sym("x");
  SX dx = SX::sym("dx");
  SX q = SX::sym("q");
  SX dq = SX::sym("dq");
  SX x_st = SX::vertcat({ x, dx, q, dq });

  SX u = SX::sym("u");

  SX ddx = -(1 / (m0 + m1 - m1 * cos(q) * cos(q))) * (L * m1 * sin(q) * dq * dq - u - m1 * g * cos(q) * sin(q));
  SX ddq = (1 / (m1 * L * cos(q) * cos(q) - L * (m0 + m1))) * (L * m1 * cos(q) * sin(q) * dq * dq - u * cos(q) - (m0 + m1) * g * sin(q));

  SX model = SX::vertcat({ dx + dT * 0.5 * ddx, ddx, dq + dT * 0.5 * ddq, ddq });

  Function f("f", { x_st, u }, { model });

  SX X = SX::sym("X", x_num, horizon + 1);
  SX U = SX::sym("U", u_num, horizon);
  SX P = SX::sym("P", x_num * 2);

  // Slice means START FROM, HOW MANY. If Slice(0,0) - empty result
  // Slice means START FROM, TO-1. If Slice(0,0) - empty result
  // X(Slice(0, x_num), 0) = P(Slice(0, x_num));

  SX obj = 0;
  SX G = 0;

  SX Q = SX::zeros(x_num, x_num);
  Q(0, 0) = 1e5;
  Q(1, 1) = 2e3;
  Q(2, 2) = 1e6;
  Q(3, 3) = 1e2;
  SX R = SX::zeros(u_num, u_num);
  R(0, 0) = 3e+1;

  SX x_act = SX::sym("x_act", x_num, 1);
  SX x_next = SX::sym("x_next", x_num, 1);
  SX x_next_euler = SX::sym("x_next", x_num, 1);
  SX u_act = SX::sym("u_act", u_num, 1);

  std::vector<SX> input;
  input.push_back(x_act);
  input.push_back(u_act);

  x_act = X(Slice(0, x_num), 0);
  G = x_act - P(Slice(0, x_num));

  for (uint32_t k = 0; k < horizon; k++)
  {
    x_act = X(Slice(0, x_num), k);
    u_act = U(Slice(0, u_num), k);
    obj = obj + mtimes((x_act - P(Slice(x_num, x_num + x_num))).T(), mtimes(Q, x_act - P(Slice(x_num, x_num + x_num)))) + mtimes(u_act.T(), mtimes(R, u_act));
    x_next = X(Slice(0, x_num), k + 1);

    input.clear();
    input.push_back(x_act);
    input.push_back(u_act);
    std::vector<SX> f_value = f(input);
    x_next_euler = x_act + dT * f_value.at(0);

    G = SX::vertcat({ G, x_next - x_next_euler });
  }

  SX OPT_variables = SX::vertcat({ SX::reshape(X, x_num * (horizon + 1), 1), SX::reshape(U, u_num * horizon, 1) });

  SXDict nlp_prob = { { "f", obj }, { "x", OPT_variables }, { "g", G }, { "p", P } };

  Dict opts;
  Dict ipopt_opts;
  opts["print_time"] = 1;
  // opts["print_time"] = 0;
  ipopt_opts["max_iter"] = 2000;
  // ipopt_opts["print_level"] = 3;
  ipopt_opts["print_level"] = 0;
  ipopt_opts["acceptable_tol"] = 1e-8;
  ipopt_opts["acceptable_obj_change_tol"] = 1e-6;
  opts["ipopt"] = ipopt_opts;

  Function solver = nlpsol("solver", "ipopt", nlp_prob, opts);

  return solver;
}

void pubPredictedState()
{
  for (uint16_t i = 0; i < num_pub_pred; i++)
  {
    sensor_msgs::JointState joint_state;

    joint_state.header.stamp = ros::Time::now();

    joint_state.header.frame_id = tf_prefix_names.at(i) + "/world";
    joint_state.name.push_back("slider_to_cart");
    joint_state.name.push_back("cart_to_stick");

    joint_state.position.clear();
    joint_state.velocity.clear();
    joint_state.effort.clear();

    joint_state.position.push_back((double)predicted_states.at(i)(0));
    joint_state.position.push_back((double)predicted_states.at(i)(1));
    joint_state.velocity.push_back(0);
    joint_state.velocity.push_back(0);
    joint_state.effort.push_back(0);
    joint_state.effort.push_back(0);
    pub_predicted_state[i].publish(joint_state);
  }
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "cart_pendulum_example_node");
  ros::NodeHandle n;
  ros::Rate rate(LOOP_RATE);

  ROS_INFO("Init");

  initPredictionPub(n);

  raisim::World::setActivationKey("~/.raisim/activation.raisim");

  raisim::World world;
  raisim::RaisimServer server(&world);
  raisim::ArticulatedSystem* robot;

  world.setTimeStep(SIM_DT);
  world.addGround();

  std::string urdf_path = "/home/splitmind/splitm_quadruped_ws/src/casadi_ipopt_examples/description/cart_pendulum_model.urdf";
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
  robot->setName("cart_pendulum");

  server.launchServer();
  server.focusOn(robot);

  Eigen::VectorXd pos_act;
  Eigen::VectorXd vel_act;
  Eigen::VectorXd u_des;
  u_des.resize(robot->getDOF());
  u_des(0) = 0;
  u_des(1) = 0;

  SX x_init = SX::sym("x_init", x_num, 1);
  SX x_des = SX::sym("x_des", x_num, 1);
  x_init(0) = 0.5;
  x_init(1) = 0;
  x_init(2) = -3.1415;
  x_init(3) = 0;
  x_des(0) = 0;
  x_des(1) = 0;
  x_des(2) = 0;
  x_des(3) = 0;

  pos_act.resize(2, 1);
  vel_act.resize(2, 1);
  pos_act(0) = (double)x_init(0);
  pos_act(1) = (double)x_init(2);
  vel_act(0) = (double)x_init(1);
  vel_act(1) = (double)x_init(3);

  robot->setState(pos_act, vel_act);

  SX U0 = SX::zeros(u_num, horizon);
  SX u = SX::zeros(horizon, u_num);
  SX X0 = repmat(x_init, 1, horizon + 1).T();

  DMDict arg;
  arg["x0"] = SX::vertcat({ SX::reshape(X0.T(), x_num * (horizon + 1), 1), SX::reshape(U0.T(), u_num * horizon, 1) });
  arg["lbg"].clear();
  arg["lbg"].resize(1, x_num * (horizon + 1));
  arg["ubg"].clear();
  arg["ubg"].resize(1, x_num * (horizon + 1));
  arg["lbg"](Slice(0, x_num * (horizon + 1))) = 0;
  arg["ubg"](Slice(0, x_num * (horizon + 1))) = 0;
  arg["lbx"] = u_min;
  arg["ubx"] = u_max;
  arg["p"] = SX::vertcat({ x_init, x_des });

  cout << "lbg size: " << arg["lbg"].size() << endl;

  Function solver = cartPendCasadiIpopt();

  DMDict res = solver(arg);

  ROS_INFO("Enter loop");

  while (ros::ok())
  {
    server.integrateWorldThreadSafe();

    robot->getState(pos_act, vel_act);

    x_init(0) = pos_act(0);
    x_init(1) = vel_act(0);
    x_init(2) = pos_act(1);
    x_init(3) = vel_act(1);
    x_des(0) = 0;
    x_des(1) = 0;
    x_des(2) = 0;
    x_des(3) = 0;

    arg["p"] = SX::vertcat({ x_init, x_des });
    arg["x0"] = SX::vertcat({ SX::reshape(X0.T(), x_num * (horizon + 1), 1), SX::reshape(U0.T(), u_num * horizon, 1) });

    res = solver(arg);

    u = reshape(res.at("x")(Slice(x_num * (horizon + 1), (int)std::vector<double>(res.at("x")).size())).T(), u_num, horizon).T();

    u_des(0) = std::vector<double>(u).at(0);
    cout << "u_des: " << u_des(0) << endl;
    robot->setGeneralizedForce(u_des);

    U0 = u;
    U0(Slice(0, horizon - 1)) = U0(Slice(1, horizon));

    X0 = reshape(res.at("x")(Slice(0, x_num * (horizon + 1))).T(), x_num, horizon + 1).T();
    X0(Slice(0, horizon)) = X0(Slice(1, horizon + 1));

    predicted_states.clear();
    predicted_states.resize(num_pub_pred);

    uint16_t horizon_part = 10;
    predicted_states.at(0) = SX::vertcat({ pos_act(0), pos_act(1) });
    predicted_states.at(1) = SX::vertcat({ (double)X0(horizon / horizon_part, 0), (double)X0(horizon / horizon_part, 2) });
    predicted_states.at(2) = SX::vertcat({ (double)X0(horizon / horizon_part * 2, 0), (double)X0(horizon / horizon_part * 2, 2) });
    predicted_states.at(3) = SX::vertcat({ (double)X0(horizon / horizon_part * 3, 0), (double)X0(horizon / horizon_part * 3, 2) });

    pubPredictedState();

    ros::spinOnce();
    rate.sleep();
  }

  server.killServer();

  return 0;
}
