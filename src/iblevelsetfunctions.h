#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>


using namespace dealii;


#ifndef LETHE_IBLEVELSETFUNCTIONS_H
#define LETHE_IBLEVELSETFUNCTIONS_H

template <int dim>
class IBLevelSetFunctions
{
public:
    IBLevelSetFunctions();
    IBLevelSetFunctions(Point<dim> p_center, Tensor<1,dim> p_linear_velocity, Tensor<1,3> p_angular_velocity):
      center(p_center),linear_velocity(p_linear_velocity),angular_velocity(p_angular_velocity)
    {}

    // Value of the distance
    virtual double distance(const Point<dim> &p) = 0;
    virtual void   velocity(const Point<dim> &p, Vector<double> &values)=0;
protected:
      Point<dim>    center;
      Tensor<1,dim> linear_velocity;
      Tensor<1,3>   angular_velocity; // rad/s
};


template <int dim>
class IBLevelSetCircle: public IBLevelSetFunctions<dim>
{
private:
  double radius;
public:
    IBLevelSetCircle(Point<dim> p_center, Tensor<1,dim> p_linear_velocity, Tensor<1,3> p_angular_velocity, double p_radius):
      IBLevelSetFunctions<dim>(p_center,p_linear_velocity,p_angular_velocity),
    radius(p_radius){}

    // Value of the distance
    virtual double distance(const Point<dim> &p)
    {
      const double x = p[0];
      const double y = p[1];
      return std::sqrt(x*x+y*y)-radius;
    }
    virtual void   velocity(const Point<dim> &p, Vector<double> &values) {}
protected:
};

//template<int dim>
//void IBLevelSetFunction<dim>::vector_value(const Point<dim> &p,
//                                           Vector<double> &values) const
//{
//    assert(dim==2);
//    const double a = M_PI;
//    const double x = p[0];
//    const double y = p[1];
//        values(0) = (2*a*a*(-sin(a*x)*sin(a*x) +
//                            cos(a*x)*(cos(a*x)))*sin(a*y)*cos(a*y)
//                     - 4*a*a*sin(a*x)*sin(a*x)*sin(a*y)*cos(a*y)
//                     - 2.0*x)*(-1.)
//                + a*std::pow(sin(a*x),3.) * std::pow(sin(a*y),2.) * std::cos(a*x);
//        values(1) = (2*a*a*(sin(a*y)*(sin(a*y)) - cos(a*y)*cos(a*y))
//                     *sin(a*x)*cos(a*x) + 4*a*a*sin(a*x)*sin(a*y)*sin(a*y)
//                     *cos(a*x) - 2.0*y)*(-1)
//                + a*std::pow(sin(a*x),2.) * std::pow(sin(a*y),3.) * std::cos(a*y);
//
//}
//
#endif
