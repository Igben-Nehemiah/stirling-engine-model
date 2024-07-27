#include "enskog.hpp"

int main(int argc, char *argv[])
{
  double density = 2.0 / (3.0 * std::sqrt(2));

  Enskog fluid(0.1, Enskog::BMCSL, 3, true, 1.0);
  fluid.addSpecies(1.0, 1.0, 1.0);
  // fluid.addSpecies(0.1, 0.001, 0.5);
  // fluid.addSpecies(1.0, 1.0, 0.5);
  // fluid.addSpecies(1.0, 0.1, 0.5);
  fluid.init();
  std::cout << "P=" << fluid.pressure() << std::endl;
  // std::cout << "L0u=" << fluid.Lau(0) << std::endl;
  // std::cout << "Lu0=" << fluid.Lua(0) << std::endl;
  // std::cout << "L1u=" << fluid.Lau(1) << std::endl;
  // std::cout << "Lu1=" << fluid.Lua(1) << std::endl;
  std::cout << "Luu=" << fluid.Luu() << std::endl;
  // std::cout << "L00=" << fluid.Lab(0,0) << std::endl;
  // std::cout << "L10=" << fluid.Lab(1,0) << std::endl;
  // std::cout << "L01=" << fluid.Lab(0,1) << std::endl;
  // std::cout << "L11=" << fluid.Lab(1,1) << std::endl;
  std::cout << "Lshear=" << fluid.shearViscosity() << std::endl;
  std::cout << "Lbulk=" << fluid.bulkViscosity() << std::endl;
}