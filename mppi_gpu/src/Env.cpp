#include "mppi_env.hpp"

ostream& operator<<(ostream& os, const Env& env)
{
    os << env.print();
    return os;
}

string Env::print() const{
  return info;
}
