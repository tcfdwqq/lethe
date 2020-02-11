#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

template<int dim>
class ExactSolutionMMS : public Function<dim>
{
public:
    ExactSolutionMMS() : Function<dim>(3) {}
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const;
};
template<int dim>
void ExactSolutionMMS<dim>::vector_value(const Point<dim> &p,
                                                    Vector<double> &values) const
{
    const double a = M_PI;
    double x = p[0];
    double y = p[1];
    values(0) = sin(a*x)*sin(a*x)*cos(a*y)*sin(a*y);
    values(1) = -cos(a*x)*sin(a*x)*sin(a*y)*sin(a*y);
    values(2) = -2 + x*x + y*y;
}

template<int dim>
class ExactSolutionTaylorCouette : public Function<dim>
{
public:
    ExactSolutionTaylorCouette() : Function<dim>(3)
    {
        eta_=ri_/ro_;
        mu=omega_2/omega_1;
        ri_=0.21;
        ro_=0.91;
        omega_1=1/ri_;
        omega_2=0;
    }
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const;

private:
    double eta_;
    double ri_=0.21;
    double ro_=0.91;
    double omega_1;
    double mu;
    double omega_2;
};
template<int dim>
void ExactSolutionTaylorCouette<dim>::vector_value(const Point<dim> &p,
                                                    Vector<double> &values) const
{
    const double a = M_PI;
    double x = p[0];
    double y = p[1];

    double r= std::sqrt(x*x+y*y);

    double theta= std::atan2(y,x);
    double A= -(eta_*eta_)  /(1.-eta_*eta_);
    double B= ri_ * ri_ / (1.-eta_*eta_);
    A= (omega_2*ro_*ro_-omega_1*ri_*ri_)/(ro_*ro_-ri_*ri_);
    B= (omega_1-omega_2)*ri_*ri_*ro_*ro_/(ro_*ro_-ri_*ri_);
    double utheta= A*r + B/r;
    if (r>ri_/eta_)
        utheta=0;
    if (r<ri_){
        utheta=omega_1*r;
    }

    values(0) = -std::sin(theta)*utheta;
    values(1) = std::cos(theta)*utheta;
    values(2) = 0.;
}
