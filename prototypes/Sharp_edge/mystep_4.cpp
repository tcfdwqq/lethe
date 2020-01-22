#include <iostream>
#include <deal.II/grid/tria.h>

#include <deal.II/grid/grid_generator.h>


#include <deal.II/base/point.h>

#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>


#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/base/point.h>
#include <deal.II/numerics/data_out.h>
using namespace dealii;
//define class that would be call during the finite elements analysis

template <int dim>
class My_step_4
{
    //Bonne pratique de limite les fonctions de types public et de regrouper le plus
    // de fonctions possible dans les fonctions privés.
    public:
        My_step_4();
        void run();

    private:
    //define all other fonction that would be call during the resolution of the problemes
        void meshing();
        void setup_matix();
        void define_probleme();
        void solve();
        void output();
        void sharp_edge();

        //define global variables

        Triangulation<dim> mesh;
        Triangulation<dim-1,dim> immersed_mesh;





        FE_Q<dim> fe;
        //FE_Q<dim-1,dim> fe_immersed;

        DoFHandler<dim> dof_handler;
        DoFHandler<dim-1,dim> dof_handler_immersed;

        //define matrix
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        FullMatrix<double> system_matrix_2;


        // define variables that  are solution of the probleme
        Vector<double> solution;
        Vector<double> system_rhs;
        Vector<double> immersed_x;
        Vector<double> immersed_y;
        Vector<double> immersed_value;

};
// define the existance of the right and side function and the boudnary valeu function  that will be evaluated during the building of the matirx
template <int dim>
class Right_hand_side : public Function <dim>{
public:
    Right_hand_side()
    :Function<dim>()
    {}
    virtual double value(const Point<dim> & p, const unsigned int component=0) const override;

};

template <int dim>
class Boundary_value : public Function<dim>
{
public:
    Boundary_value()
    :Function<dim>()
    {}
    virtual double value(const Point<dim> & p,
            const unsigned int component=0) const override;

};


template <int dim>
double Right_hand_side <dim>:: value(const Point<dim> &p,const unsigned int /*component*/) const
{
    double return_value=0.0;
    for (unsigned int i =0 ; i< dim;++i)
        return_value+=4*(std::pow(p(i),4.0));
    return 0;
    //return return_value;

}



template <int dim>
double Boundary_value <dim> :: value(const Point<dim> &p, const unsigned int /*component*/ ) const{
    return 0;
    //return p.square();
}


// define the type of elements we want to used ( linear) in the finite elements methode
// also associate the triangulation object to the dof handler
template <int dim>
My_step_4<dim>::My_step_4()
    : fe(1)
    // define dof from mesh
    , dof_handler(mesh)
    ,dof_handler_immersed(immersed_mesh){};

template <int dim>
void My_step_4<dim>::meshing()
    {
    // first step is to used the triangulation object to creat a space whit the fonction make grid
    const Point<2> center_immersed(0,0);
    GridGenerator::hyper_ball(mesh,center_immersed,0.8);
    // raffining the basic mesh that we got
    mesh.refine_global(4);
    std::cout<<"Number of active cells: "<< mesh.n_active_cells()<<std::endl;
    }

template <int dim>
void My_step_4<dim>::setup_matix() {
    //define finite elements from dof
    dof_handler.distribute_dofs(fe);

    //define potentiel non zero for pattern
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    // make temporary sparsity pattern
    DoFTools::make_sparsity_pattern(dof_handler,dsp);
    //creat a copy of the dynamics sparsity parttern to used the be used on the systeme matrix
    sparsity_pattern.copy_from(dsp);
    // define where are the non zero in the matrix from sparsity pattern
    system_matrix.reinit(sparsity_pattern);
    // define de solution and right and side of the equation size
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());


    // define the immersed boundary points
    unsigned int nb_immersed=10000;
    immersed_x.reinit(nb_immersed);
    immersed_y.reinit(nb_immersed);
    immersed_value.reinit(nb_immersed);
    using numbers::PI;
    const double center_x=0;
    const double center_y=0;


    const Point<2> center_immersed(center_x,center_y);
    double radius=0.21;
    double radius_2=0.61;

    for (unsigned int i=0 ;i <nb_immersed/2;++i){
        immersed_x(i)=radius*cos(i*2*PI/(nb_immersed/2))+center_x;
        immersed_y(i)=radius*sin(i*2*PI/(nb_immersed/2))+center_y;
        immersed_value(i)=1;
    }
    for (unsigned int i=nb_immersed/2 ;i <nb_immersed;++i){
        immersed_x(i)=radius_2*cos(i*2*PI/(nb_immersed/2))+center_x;
        immersed_y(i)=radius_2*sin(i*2*PI/(nb_immersed/2))+center_y;
        immersed_value(i)=0;
    }
    std::cout<<"x "<< immersed_x<<std::endl;
    std::cout<<"y "<< immersed_y<<std::endl;
    GridGenerator::hyper_cube(immersed_mesh,0,1);
    immersed_mesh.refine_global(12);
    //dof_handler_immersed.distribute_dofs(fe_immersed);

    std::cout<<"Number of active cells: "<< immersed_mesh.n_active_cells()<<std::endl;
    //immersed_value.reinit(dof_handler_immersed.n_dofs());

    //for (unsigned int i=0 ;i <dof_handler_immersed.n_dofs();++i){
       // immersed_value(i)=1;
    //}

}



template <int dim>
void My_step_4<dim>::sharp_edge() {
// overwrite the line for the point in mesh
    MappingQ1<dim> immersed_map;
    std::map< types::global_dof_index, Point< dim >>  	support_points;
    DoFTools::map_dofs_to_support_points(immersed_map,dof_handler,support_points);

    QGauss<dim> q_formula(fe.degree+1);
    // we need to define what part of the finite elements we need to compute in orther to solve the equation we want
    // in or case wee need the gradient of the shape function the jacobians of the matrix and the shape function values
    FEValues<dim> fe_values(fe, q_formula,update_quadrature_points);
    FEValues<dim> fe_values_2(fe, q_formula,update_quadrature_points);
    const unsigned int n_q_points = q_formula.size();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_2(dofs_per_cell);

    unsigned int best_vertex = 0;



    std::vector<Point<dim>> support_point(dof_handler.n_dofs());
    const auto &cell_iterator=dof_handler.active_cell_iterators();

    double min_cell_d=(GridTools::minimal_cell_diameter(mesh)*GridTools::minimal_cell_diameter(mesh))/sqrt(2*(GridTools::minimal_cell_diameter(mesh)*GridTools::minimal_cell_diameter(mesh)));



    using numbers::PI;
    const double center_x=0;
    const double center_y=0;

    const Point<2> center_immersed(center_x,center_y);
    double radius=0.2;
    double radius_2=0.4;




    for (const auto &cell : cell_iterator)  {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            Point<dim> vertices_ib_j(immersed_x(0),immersed_y(0));
            double       best_dist_ib   = sqrt((support_points[local_dof_indices[q_point]]- vertices_ib_j).norm_square());
            double       ib_value_select=1;
            Tensor<1,2, double> best_vect_dist = (support_points[local_dof_indices[q_point]] - vertices_ib_j);
            for (unsigned int j = 0; j < immersed_x.size(); j++) {
                    Point<dim> vertices_ib_j(immersed_x(j),immersed_y(j));

                    Tensor<1,2, double> vect_dist = (support_points[local_dof_indices[q_point]] - vertices_ib_j);
                    double dist=sqrt(vect_dist[1]*vect_dist[1]+vect_dist[0]*vect_dist[0]);
                    //Tensor<1,2 < double>>
                    if (dist < best_dist_ib) {
                        best_vect_dist=vect_dist;

                        best_dist_ib = dist;
                        ib_value_select=immersed_value(j);
                    }
                }

                if (best_dist_ib<min_cell_d ) {

                    unsigned int global_index_overrigth=local_dof_indices[q_point];
                    std::cout << "best dist: " << best_dist_ib << std::endl;
                    std::cout << "position of dof: " << support_points[local_dof_indices[q_point]] << std::endl;
                    std::cout << "index global of dof: " << local_dof_indices[q_point] << std::endl;
                    if(best_dist_ib!=0){

                    const Point<dim> second_point(support_points[local_dof_indices[q_point]]+best_vect_dist);
                    const auto &cell_2=GridTools::find_active_cell_around_point(dof_handler,second_point);



                    cell_2->get_dof_indices(local_dof_indices);
                    Point<dim> second_point_v = immersed_map.transform_real_to_unit_cell(cell_2,second_point);


                    for (unsigned int j =  0; j < dof_handler.n_dofs(); j++)
                        system_matrix.set(global_index_overrigth,j,0);

                    system_matrix.add(global_index_overrigth,global_index_overrigth,-2/(best_dist_ib*best_dist_ib) );

                    for (unsigned int j =  0; j < fe.dofs_per_cell; j++) {

                           system_matrix.add(global_index_overrigth, local_dof_indices[j],
                                              fe.shape_value(j, second_point_v)/(best_dist_ib*best_dist_ib));
                    }
                    system_rhs(global_index_overrigth)=-ib_value_select/(best_dist_ib*best_dist_ib);

                }
                else{
                        for (unsigned int j =  0; j < dof_handler.n_dofs(); j++)
                            system_matrix.set(global_index_overrigth,j,0);

                        system_matrix.add(global_index_overrigth,global_index_overrigth,1 );
                        system_rhs(global_index_overrigth)=0;

                }
            }
        }
    }
}

template <int dim>
void My_step_4<dim>::define_probleme() {
    //define quadrature formula for the integral on choisie un ordre de quadrature 1 superieur au element
    QGauss<dim> q_formula(fe.degree + 1);
    // we need to define what part of the finite elements we need to compute in orther to solve the equation we want
    // in or case wee need the gradient of the shape function the jacobians of the matrix and the shape function values

    FEValues<dim> fe_values(fe, q_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points);

    // define shortcut for iteration ( presented in the code )


    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = q_formula.size();


    //define the full matrix of a single cell and vector for the resolution.

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    // define a vector whit the position in matrix of every DOf  of th local matrix
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    //define the righthand side value of the equation
    const Right_hand_side<dim> righthandside;

//loop over all cell to creat small matrix and define the contribution of each cell

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        //recalcule les valeurs pour la cellule d'intéret

        fe_values.reinit(cell);
        //initialise les valeurs de la cellule local pour ne pas transporté les valeur de la cellule précedente.
        cell_matrix = 0;
        cell_rhs = 0;
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            for (unsigned int i=0; i < dofs_per_cell; ++i)
                for (unsigned int j=0; j < dofs_per_cell; ++j)
                    //add values to the cell matrix the matrix is somme over all point of grad_phi_i*grad_phi_j*jacobian matrix
                    cell_matrix(i, j) += fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);


            for (unsigned int i=0; i<dofs_per_cell;++i) {
                //right and side of the function whit f(x)=1 so phi*righthandside*jacobian
                const auto x_q = fe_values.quadrature_point(q_point);
                cell_rhs(i) = fe_values.shape_value(i, q_point) * righthandside.value(x_q) * fe_values.JxW(q_point);
            }
        }

        // define where the celle information we calculated must go in the global matrix
        cell->get_dof_indices(local_dof_indices);

        // calculate the local contribution on the global matrix of the cell matrix
        for (unsigned int i=0 ;i <dofs_per_cell;++i)
            for (unsigned int j=0 ;j <dofs_per_cell;++j)
                system_matrix.add(local_dof_indices[i],local_dof_indices[j],cell_matrix(i,j));

        // same for the rhs

        for (unsigned int i=0 ;i <dofs_per_cell;++i)
            system_rhs(local_dof_indices[i])+=cell_rhs(i);
    }
    // setting in place the boundary conditions in the cae that we want we nee to fix all boundary  whit a dirichlet =0

    // the mapping of boudnary condition whit the DOF
    std::map<types::global_dof_index, double> boundary_values;

    // setting the boudary value trough interpolation

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Boundary_value<dim>(),
                                             boundary_values);
    // now we nee to implement the boundary value  to the matrix and the right hand side of the equation
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);
}
template <int dim>
void My_step_4<dim>::solve() {
    // setting the solver iteration and tolerance
    SolverControl solver_control(10000,1e-12);
    //setting the solve in place
    SolverCG<> solver (solver_control);
    // apply the solver to the probleme

    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult(solution, system_rhs);
    //solver.solve(system_matrix,solution,system_rhs,PreconditionIdentity());

}
template <int dim>
void My_step_4<dim>::output() {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,"solution");
    data_out.build_patches();

    std::ofstream output("solution.vtk");
    data_out.write_vtk(output);

    /*DataOut<dim> data_out_2;
    data_out_2.attach_dof_handler(dof_handler_immersed);
    data_out_2.add_data_vector(immersed_value,"immersed_value");
    data_out_2.build_patches();
    std::ofstream output_2("immersed.vtk");
    data_out.write_vtk(output_2);*/
}
template <int dim>
void My_step_4<dim>::run(){
    // Call function in sequence
    meshing();
    setup_matix();
    define_probleme();
    sharp_edge();
    solve();
    output();


}

int main()
// call log information of the solving process
{deallog.depth_console(2);
// give the class a proper function name  and then run it
My_step_4<2> laplace;

laplace.run();
return 0;
}

