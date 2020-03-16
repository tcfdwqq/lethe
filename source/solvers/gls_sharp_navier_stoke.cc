/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - by the Lethe authors
 *
 * This file is part of the Lethe library
 *
 * The Lethe library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 3.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the Lethe distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Bruno Blais, Polytechnique Montreal, 2019-
 */

#include "solvers/gls_sharp_navier_stokes.h"

#include "core/sdirk.h"

// Constructor for class GLSNavierStokesSolver
template <int dim>
GLSNavierStokesSharpSolver<dim>::GLSNavierStokesSharpSolver(
  NavierStokesSolverParameters<dim> &p_nsparam,
  const unsigned int                 p_degreeVelocity,
  const unsigned int                 p_degreePressure)
  : NavierStokesBase<dim, TrilinosWrappers::MPI::Vector, IndexSet>(
      p_nsparam,
      p_degreeVelocity,
      p_degreePressure)

{

}

template <int dim>
GLSNavierStokesSharpSolver<dim>::~GLSNavierStokesSharpSolver()
{
  this->dof_handler.clear();
}

template <int dim>
void GLSNavierStokesSharpSolver<dim>::vertices_cell_mapping()
{
    //map the vertex index to the cell that include that vertex used later in which cell a point falls in
    //vertices_to_cell is a vector of vectof of dof handler active cell iterator each element i of the vector is a vector of all the cell in contact with the vertex i
    //std::cout << "this MPI porcess start vertex mapping : "<< this->this_mpi_process<< std::endl;
    vertices_to_cell.clear();
    vertices_to_cell.resize(this->dof_handler.n_dofs()/(dim+1));
    const auto &cell_iterator=this->dof_handler.active_cell_iterators();
    //loop on all the cell and
    for (const auto &cell : cell_iterator) {
        if (cell->is_locally_owned()| cell->is_ghost()) {
            unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;
            for (unsigned int i = 0; i < vertices_per_cell; i++) {
                //add this cell as neighbors for all it's vertex
                unsigned int v_index = cell->vertex_index(i);
                std::vector<typename DoFHandler<dim>::active_cell_iterator> adjacent = vertices_to_cell[v_index];
                //can only add the cell if it's a set and not a vector
                std::set<typename DoFHandler<dim>::active_cell_iterator> adjacent_2(adjacent.begin(), adjacent.end());
                adjacent_2.insert(cell);
                //convert back the set to a vector and add it in the vertices_to_cell;
                std::vector<typename DoFHandler<dim>::active_cell_iterator> adjacent_3(adjacent_2.begin(),
                                                                                       adjacent_2.end());
                vertices_to_cell[v_index] = adjacent_3;
            }
        }
    }
    //std::cout << "this MPI porcess finish vertex mapping : "<< this->this_mpi_process<< std::endl;
}

template <int dim>
void GLSNavierStokesSharpSolver<dim>::define_particules() {
    //define position and velocity of particules
        particules.resize(this->nsparam.particulesParameters.nb);
        // define position of particules
        //x y z
        if (dim ==2) {
            for (unsigned int i=0 ; i< this->nsparam.particulesParameters.nb;++i) {
                particules[i].resize(3 * dim);
                //x y
                particules[i][0] = this->nsparam.particulesParameters.particules[i][0];
                particules[i][1] = this->nsparam.particulesParameters.particules[i][1];
                //Vx Vy
                particules[i][2] = this->nsparam.particulesParameters.particules[i][3];
                particules[i][3] = this->nsparam.particulesParameters.particules[i][4];
                //omega
                particules[i][4] = this->nsparam.particulesParameters.particules[i][8];
                //radius
                particules[i][5] = this->nsparam.particulesParameters.particules[i][9];;
            }
        }

        if (dim ==3) {
            for (unsigned int i=0 ; i< this->nsparam.particulesParameters.nb;++i) {
                particules[i].resize(3 * dim+1);

                //x y
                particules[i][0] = this->nsparam.particulesParameters.particules[i][0];
                particules[i][1] = this->nsparam.particulesParameters.particules[i][1];
                particules[i][2] = this->nsparam.particulesParameters.particules[i][2];

                //Vx Vy
                particules[i][3] = this->nsparam.particulesParameters.particules[i][3];
                particules[i][4] = this->nsparam.particulesParameters.particules[i][4];
                particules[i][5] = this->nsparam.particulesParameters.particules[i][5];
                //omega
                particules[i][6] = this->nsparam.particulesParameters.particules[i][6];
                particules[i][7] = this->nsparam.particulesParameters.particules[i][7];
                particules[i][8] = this->nsparam.particulesParameters.particules[i][8];
                //radius
                particules[i][9] = this->nsparam.particulesParameters.particules[i][9];;
            }
        }
}

template <int dim>
void GLSNavierStokesSharpSolver<dim>::clear_pressure() {
    for(unsigned int i=dim;i < this->dof_handler.n_dofs(); i+=dim+1){
    //for(unsigned int i=this->dof_handler.n_dofs()*dim/(dim+1);i < this->dof_handler.n_dofs(); ++i){
            if (this->locally_owned_dofs.is_element(i)){
                this->present_solution(i)=0;
            }

        }


}




template <int dim>
void GLSNavierStokesSharpSolver<dim>::force_on_ib() {
    // cumpute the torque for a couet flow on the immersed boundary
    if (dim==2) {
        for (unsigned int p = 0; p < particules.size(); ++p) {
            using numbers::PI;
            const double center_x = particules[p][0];
            const double center_y = particules[p][1];


            QGauss<dim> q_formula(this->fe.degree + 1);
            FEValues<dim> fe_values(this->fe, q_formula, update_quadrature_points);

            double mu = this->nsparam.physicalProperties.viscosity;

            MappingQ1<dim> immersed_map;
            std::vector<types::global_dof_index> local_dof_indices(this->fe.dofs_per_cell);
            std::vector<types::global_dof_index> local_dof_indices_2(this->fe.dofs_per_cell);
            std::vector<types::global_dof_index> local_dof_indices_3(this->fe.dofs_per_cell);
            unsigned int nb_evaluation = 100;
            double t_torque = 0;

            double fx_v = 0;
            double fy_v = 0;

            double fx_p_0 = 0;
            double fy_p_0 = 0;
            double fx_p_1 = 0;
            double fy_p_1 = 0;
            double fx_p_2 = 0;
            double fy_p_2 = 0;

            double dr = (GridTools::minimal_cell_diameter(*this->triangulation) *
                         GridTools::minimal_cell_diameter(*this->triangulation)) / sqrt(2 *
                                                                                        (GridTools::minimal_cell_diameter(
                                                                                                *this->triangulation) *
                                                                                         GridTools::minimal_cell_diameter(
                                                                                                 *this->triangulation)));
            dr = dr * 2;
            for (unsigned int i = 0; i < nb_evaluation; ++i) {
                const Point<dim> eval_point(particules[p][5] * cos(i * 2 * PI / (nb_evaluation)) + center_x,
                                            particules[p][5] * sin(i * 2 * PI / (nb_evaluation)) + center_y);
                const auto &cell = GridTools::find_active_cell_around_point(this->dof_handler, eval_point);
                if (cell->is_locally_owned()) {

                    cell->get_dof_indices(local_dof_indices);
                    double u_1 = 0;
                    double v_1 = 0;
                    u_1 = -particules[p][4]*particules[p][5]*sin(i * 2 * PI / (nb_evaluation))+particules[p][2];
                    v_1 = particules[p][4]*particules[p][5]*cos(i * 2 * PI / (nb_evaluation))+particules[p][3];
                    double U1 =
                            u_1 * cos(i * 2 * PI / (nb_evaluation) - PI / 2) +
                            v_1 * sin(i * 2 * PI / (nb_evaluation) - PI / 2);

                    const Point<dim> eval_point_2(eval_point[0] + dr * cos(i * 2 * PI / (nb_evaluation)),
                                                  eval_point[1] + dr * sin(i * 2 * PI / (nb_evaluation)));
                    const auto &cell_2 = GridTools::find_active_cell_around_point(this->dof_handler, eval_point_2);
                    Point<dim> second_point_v = immersed_map.transform_real_to_unit_cell(cell_2, eval_point_2);
                    cell_2->get_dof_indices(local_dof_indices);
                    double u_2 = 0;
                    double v_2 = 0;
                    for (unsigned int j = 0; j < 12; j = j + 3) {
                        u_2 += this->fe.shape_value(j, second_point_v) * this->present_solution(local_dof_indices[j]);
                        v_2 += this->fe.shape_value(j + 1, second_point_v) *
                               this->present_solution(local_dof_indices[j + 1]);
                    }
                    double U2 =
                            u_2 * cos(i * 2 * PI / (nb_evaluation) - PI / 2) +
                            v_2 * sin(i * 2 * PI / (nb_evaluation) - PI / 2);
                    double du_dr = (U2 / (particules[p][5] + dr) - U1 / particules[p][5]) / dr;

                    double local_fx_v=du_dr * mu * particules[p][5] * 2 * PI * particules[p][5] / (nb_evaluation - 1) *
                                      sin(i * 2 * PI / (nb_evaluation));
                    double local_fy_v=-du_dr * mu * particules[p][5] * 2 * PI * particules[p][5] / (nb_evaluation - 1) *
                                      cos(i * 2 * PI / (nb_evaluation));

                    fx_v += local_fx_v;
                    fy_v += local_fy_v;
                    t_torque += local_fx_v * sin(i * 2 * PI / (nb_evaluation) )* particules[p][5]-local_fy_v*cos(i * 2 * PI / (nb_evaluation))* particules[p][5];
                }
            }

            double t_torque_ =Utilities::MPI::sum(t_torque, this->mpi_communicator);

            if  (this->this_mpi_process == 0){
                std::cout <<"particule : "<< p << " total_torque :" << t_torque_ << std::endl;
            }

            fx_p_0 = 0;
            fy_p_0 = 0;
            fx_p_1 = 0;
            fy_p_1 = 0;
            fx_p_2 = 0;
            fy_p_2 = 0;
            for (unsigned int i = 0; i < nb_evaluation; ++i) {

                const Point<dim> eval_point(particules[p][5] * cos(i * 2 * PI / (nb_evaluation)) + center_x,
                                            particules[p][5] * sin(i * 2 * PI / (nb_evaluation)) + center_y);
                const Point<dim> eval_point_2(eval_point[0] + 1 * dr * cos(i * 2 * PI / (nb_evaluation)),
                                              eval_point[1] + 1 * dr * sin(i * 2 * PI / (nb_evaluation)));
                const Point<dim> eval_point_3(eval_point[0] + 2 * dr * cos(i * 2 * PI / (nb_evaluation)),
                                              eval_point[1] + 2 * dr * sin(i * 2 * PI / (nb_evaluation)));
                const Point<dim> eval_point_4(eval_point[0] + 3 * dr * cos(i * 2 * PI / (nb_evaluation)),
                                              eval_point[1] + 3 * dr * sin(i * 2 * PI / (nb_evaluation)));
                const auto &cell = GridTools::find_active_cell_around_point(this->dof_handler, eval_point_2);
                if (cell->is_locally_owned()) {
                    const auto &cell2 = GridTools::find_active_cell_around_point(this->dof_handler, eval_point_3);
                    const auto &cell3 = GridTools::find_active_cell_around_point(this->dof_handler, eval_point_4);

                    Point<dim> second_point_v = immersed_map.transform_real_to_unit_cell(cell, eval_point_2);
                    Point<dim> second_point_v_2 = immersed_map.transform_real_to_unit_cell(cell2, eval_point_3);
                    Point<dim> second_point_v_3 = immersed_map.transform_real_to_unit_cell(cell3, eval_point_4);
                    cell->get_dof_indices(local_dof_indices);

                    cell2->get_dof_indices(local_dof_indices_2);
                    cell3->get_dof_indices(local_dof_indices_3);
                    double P_1 = 0;
                    double P_2 = 0;
                    double P_3 = 0;
                    for (unsigned int j = 2; j < 12; j = j + 3) {
                        P_1 += this->fe.shape_value(j, second_point_v) * this->present_solution(local_dof_indices[j]);
                        P_2 += this->fe.shape_value(j, second_point_v_2) *
                               this->present_solution(local_dof_indices_2[j]);
                        P_3 += this->fe.shape_value(j, second_point_v_3) *
                               this->present_solution(local_dof_indices_3[j]);
                    }
                    double P2_temp = P_1 + (P_1 - P_2) + ((P_1 - P_2) - (P_2 - P_3));
                    double P2 = P2_temp;//+(P2_temp-P_1)+((P2_temp-P_1)-(P_1-P_2));
                    double P = P_1 + (P_1 - P_2) * 1;
                    double P3 = P_1;
                    fx_p_2 += P2 * -cos(i * 2 * PI / (nb_evaluation)) * 2 * PI * particules[p][5] / (nb_evaluation - 1);
                    fy_p_2 += P2 * -sin(i * 2 * PI / (nb_evaluation)) * 2 * PI * particules[p][5] / (nb_evaluation - 1);
                    fx_p_1 += P * -cos(i * 2 * PI / (nb_evaluation)) * 2 * PI * particules[p][5] / (nb_evaluation - 1);
                    fy_p_1 += P * -sin(i * 2 * PI / (nb_evaluation)) * 2 * PI * particules[p][5] / (nb_evaluation - 1);
                    fx_p_0 += P3 * -cos(i * 2 * PI / (nb_evaluation)) * 2 * PI * particules[p][5] / (nb_evaluation - 1);
                    fy_p_0 += P3 * -sin(i * 2 * PI / (nb_evaluation)) * 2 * PI * particules[p][5] / (nb_evaluation - 1);

                }
            }
            double fx_p_2_ =Utilities::MPI::sum(fx_p_2, this->mpi_communicator);
            double fy_p_2_ =Utilities::MPI::sum(fy_p_2, this->mpi_communicator);
            double fx_p_1_ =Utilities::MPI::sum(fx_p_1, this->mpi_communicator);
            double fy_p_1_ =Utilities::MPI::sum(fy_p_1, this->mpi_communicator);
            double fx_p_0_ =Utilities::MPI::sum(fx_p_0, this->mpi_communicator);
            double fy_p_0_ =Utilities::MPI::sum(fy_p_0, this->mpi_communicator);
            double fx_v_ =Utilities::MPI::sum(fx_v, this->mpi_communicator);
            double fy_v_ =Utilities::MPI::sum(fy_v, this->mpi_communicator);
            if  (this->this_mpi_process == 0){
            std::cout << "ordre 0 fx_P: " << fx_p_0_ << std::endl;
            std::cout << "ordre 0 fy_P: " << fy_p_0_ << std::endl;
            std::cout << "ordre 1 fx_P: " << fx_p_1_ << std::endl;
            std::cout << "ordre 1 fy_P: " << fy_p_1_ << std::endl;
            std::cout << "ordre 2 fx_P: " << fx_p_2_ << std::endl;
            std::cout << "ordre 2 fy_P: " << fy_p_2_ << std::endl;
            std::cout << "fx_v: " << fx_v_ << std::endl;
            std::cout << "fy_v: " << fy_v_ << std::endl;
            }

        }
    }
}

template <int dim>
void GLSNavierStokesSharpSolver<dim>::refine_ib() {
    Point<dim> center_immersed;
    MappingQ1<dim> immersed_map;
    std::map< types::global_dof_index, Point< dim >>  	support_points;
    DoFTools::map_dofs_to_support_points(immersed_map,this->dof_handler,support_points);
    const unsigned int dofs_per_cell = this->fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    //define cell iterator
    const auto &cell_iterator=this->dof_handler.active_cell_iterators();
    for (const auto &cell : cell_iterator) {
        if (cell->is_locally_owned()) {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int p = 0; p < particules.size(); ++p) {
                unsigned int count_small = 0;
                if (dim == 2) {
                    center_immersed(0) = particules[p][0];
                    center_immersed(1) = particules[p][1];

                } else if (dim == 3) {
                    center_immersed(0) = particules[p][0];
                    center_immersed(1) = particules[p][1];
                    center_immersed(2) = particules[p][2];

                }
                for (unsigned int j = 0; j < local_dof_indices.size(); ++j) {
                    //count the number of dof that ar smaller or larger then the radius of the particules
                    //if all the dof are on one side the cell is not cut by the boundary meaning we dont have to do anything
                    if ((support_points[local_dof_indices[j]] - center_immersed).norm() <=
                        particules[p][particules[p].size() - 1] * this->nsparam.particulesParameters.outside_radius &
                        (support_points[local_dof_indices[j]] - center_immersed).norm() >=
                        particules[p][particules[p].size() - 1] * this->nsparam.particulesParameters.inside_radius) {
                        ++count_small;
                    }
                }
                if (count_small > 0) {
                    cell->set_refine_flag();
                }
            }
        }
    }
}


template <int dim>
void GLSNavierStokesSharpSolver<dim>::sharp_edge(const bool initial_step) {
    //This function define a immersed boundary base on the sharp edge method on a hyper_shere of dim 2 or 3

    //define stuff  in a later version the center of the hyper_sphere would be defined by a particule handler and the boundary condition associeted with it also.
    using numbers::PI;
    Point<dim> center_immersed;
    Point<dim> pressure_bridge;
    std::vector<typename DoFHandler<dim>::active_cell_iterator> active_neighbors;

    Vector <int> dof_done;
    dof_done.reinit(this->dof_handler.n_dofs());
    //define a map to all dof and it's support point
    MappingQ1<dim> immersed_map;
    std::map< types::global_dof_index, Point< dim >>  	support_points;
    DoFTools::map_dofs_to_support_points(immersed_map,this->dof_handler,support_points);

    // initalise fe value object in order to do calculation with it later
    QGauss<dim> q_formula(this->degreeQuadrature_);
    FEValues<dim> fe_values(this->fe, q_formula,update_quadrature_points|update_JxW_values);
    const unsigned int dofs_per_cell = this->fe.dofs_per_cell;

    unsigned int n_q_points  = q_formula.size();
    // define multiple local_dof_indices one for the cell iterator one for the cell with the second point for
    // the sharp edge stancil and one for manipulation on the neighbors cell.
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_2(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_3(dofs_per_cell);

    //define cell iterator
    const auto &cell_iterator=this->dof_handler.active_cell_iterators();


    //loop on all the cell to define if the sharp edge cut them
    for (const auto &cell : cell_iterator) {
        if (cell->is_locally_owned()) {
            double sum_line=0 ;
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            std::vector<int> set_pressure_cell;
            set_pressure_cell.resize(particules.size());
            // define the order of magnetude for the stencil.
            for(unsigned int qf =0 ; qf<n_q_points  ; ++qf) {
                    sum_line += fe_values.JxW(qf)*10;
            }
            // loop over all particule  to see if one of them is cutting this cell
            for (unsigned int p = 0; p < particules.size(); ++p) {
                unsigned int count_small = 0;
                if (dim == 2) {
                    center_immersed(0) = particules[p][0];
                    center_immersed(1) = particules[p][1];
                    // define arbitrary point on the boundary where the pressure will be link between the 2 domain
                    pressure_bridge(0) = particules[p][0]-this->nsparam.particulesParameters.pressure_offset[p][0];
                    pressure_bridge(1) = particules[p][1]-this->nsparam.particulesParameters.pressure_offset[p][1];
                }
                else if (dim == 3) {
                    center_immersed(0) = particules[p][0];
                    center_immersed(1) = particules[p][1];
                    center_immersed(2) = particules[p][2];
                    // define arbitrary point on the boundary where the pressure will be link between the 2 domain
                    pressure_bridge(0) = particules[p][0]-this->nsparam.particulesParameters.pressure_offset[p][0];
                    pressure_bridge(1) = particules[p][1]-this->nsparam.particulesParameters.pressure_offset[p][1];
                    pressure_bridge(2) = particules[p][2]-this->nsparam.particulesParameters.pressure_offset[p][2];

                }

                for (unsigned int j = 0; j < local_dof_indices.size(); ++j) {
                    //count the number of dof that ar smaller or larger then the radius of the particules
                    //if all the dof are on one side the cell is not cut by the boundary meaning we dont have to do anything
                    if ((support_points[local_dof_indices[j]] - center_immersed).norm() <= particules[p][particules[p].size()-1]) {
                        ++count_small;
                    }
                }

                // impose the pressure inside the particule if the inside of the particule is solved
                if(this->nsparam.particulesParameters.assemble_inside & this->nsparam.particulesParameters.P_assemble==Parameters::Particule_Assemble_type::NS ) {
                        bool cell_found = false;
                        try {
                            //define the cell and check if the point is inside of the cell
                            const Point<dim, double> p_cell = immersed_map.transform_real_to_unit_cell(
                                    cell, pressure_bridge);
                            const double dist_2 = GeometryInfo<dim>::distance_to_unit_cell(p_cell);

                            //define the cell and check if the point is inside of the cell
                            if (dist_2 == 0) {
                                //if the point is in this cell then the dist is equal to 0 and we have found our cell
                                cell_found = true;
                                //std::cout << "Pressure bridge cell found " << std::endl;
                            }
                        }
                            // may cause error if the point is not in cell
                        catch (typename MappingQGeneric<dim>::ExcTransformationFailed) {
                        }

                        if (cell_found) {
                            unsigned int inside_index = local_dof_indices[dim];
                            //loop over all the pressure point in the cell and impose the pressure on one dof inside to be equal to the pressure of one dof outside
                            for (unsigned int m = 0; m < this->dof_handler.n_dofs(); m++) {
                                this->system_matrix.set(inside_index, m, 0);
                            }
                            system_matrix.set(inside_index, local_dof_indices[dim], sum_line);
                            /*for (unsigned int j = dim; j < local_dof_indices.size(); j += dim + 1) {
                                    system_matrix.set(inside_index, local_dof_indices[j], sum_line);
                            }*/
                            /*std::cout << "Pressure_bridge dof : " << inside_index << std::endl;
                            std::cout << "Pressure_bridge dof : " << support_points[inside_index] << std::endl;
                            std::cout << "pressure_bridge : " << pressure_bridge << std::endl;*/


                            if (initial_step)
                                this->system_rhs(inside_index) = 0*sum_line;
                            else
                                this->system_rhs(inside_index) = 0;

                        }

                }


                //if the cell is cut by the IB the count wont equal 0 or the number of total dof in a cell

                if (count_small != 0 and count_small != local_dof_indices.size()) {
                    //if we are here the cell is cut by the immersed boundary
                    //loops on the dof that reprensant the velocity   in x and y and pressure separatly
                    for (unsigned int k = 0; k < dim + 1; ++k) {

                        if (k < dim) {
                            //we are working on the velocity of th
                            //loops on the dof that are for vx or vy separatly

                            for (unsigned int l = k; l < local_dof_indices.size(); l += dim + 1) {

                                if (dof_done(local_dof_indices[l])==0) {
                                    dof_done(local_dof_indices[l])+=1;
                                    // define which dof is going to be redefine
                                    unsigned int global_index_overrigth = local_dof_indices[l];

                                    //define the distance vector between the immersed boundary and the dof support point for each dof
                                    Tensor<1, dim, double> vect_dist = (support_points[local_dof_indices[l]] -
                                                                        center_immersed - particules[p][particules[p].size()-1]  *
                                                                                          (support_points[local_dof_indices[l]] -
                                                                                           center_immersed) /
                                                                                          (support_points[local_dof_indices[l]] -
                                                                                           center_immersed).norm());


                                    //define the other point for or 3 point stencil ( IB point, original dof and this point)
                                    const Point<dim, double> second_point(
                                            support_points[local_dof_indices[l]] + vect_dist);
                                    const Point<dim, double> third_point(
                                            support_points[local_dof_indices[l]] + vect_dist/2);

                                    //define the vertex associated with the dof
                                    unsigned int v = floor(l / (dim + 1));
                                    unsigned int v_index = cell->vertex_index(v);

                                    //get a cell iterator for all the cell neighbors of that vertex
                                    active_neighbors = this->vertices_to_cell[v_index];

                                    unsigned int cell_found = 0;
                                    unsigned int n_active_cells = active_neighbors.size();

                                    //loops on those cell to find in which of them the new point for or sharp edge stencil is
                                    for (unsigned int cell_index = 0; cell_index < n_active_cells; ++cell_index) {
                                        try {
                                            //define the cell and check if the point is inside of the cell
                                            const Point<dim, double> p_cell = immersed_map.transform_real_to_unit_cell(
                                                    active_neighbors[cell_index], second_point);
                                            const Point<dim, double> p_cell_2 = immersed_map.transform_real_to_unit_cell(
                                                    active_neighbors[cell_index], third_point);
                                            const double dist_2 = GeometryInfo<dim>::distance_to_unit_cell(p_cell);
                                            const double dist_3 = GeometryInfo<dim>::distance_to_unit_cell(p_cell_2);

                                            //define the cell and check if the point is inside of the cell
                                            if (dist_2 == 0 & dist_3 == 0 ) {
                                                //if the point is in this cell then the dist is equal to 0 and we have found our cell
                                                cell_found = cell_index;
                                                break;
                                            }
                                        }
                                            // may cause error if the point is not in cell
                                        catch (typename MappingQGeneric<dim>::ExcTransformationFailed) {
                                        }
                                    }

                                    //we have or next cell need to complet the stencil and we define stuff around it
                                    const auto &cell_2 = active_neighbors[cell_found];

                                    //define the unit cell point for the 3rd point of our stencil for a interpolation
                                    Point<dim> second_point_v = immersed_map.transform_real_to_unit_cell(cell_2,
                                                                                                         second_point);

                                    Point<dim> third_point_v = immersed_map.transform_real_to_unit_cell(
                                            cell_2,
                                            third_point);
                                    cell_2->get_dof_indices(local_dof_indices_2);

                                    //clear the current line of this dof  by looping on the neighbors cell of this dof and clear all the associated dof
                                    for (unsigned int m = 0; m < active_neighbors.size(); m++) {
                                        const auto &cell_3 = active_neighbors[m];
                                        cell_3->get_dof_indices(local_dof_indices_3);
                                        for (unsigned int o = 0; o < local_dof_indices_3.size(); ++o) {
                                            this->system_matrix.set(global_index_overrigth, local_dof_indices_3[o], 0);
                                        }
                                    }
                                    //define the new matrix entry for this dof
                                    if (true) {
                                        // first the dof itself

                                        for (unsigned int n = k; n < local_dof_indices.size(); n += dim + 1) {
                                            // first the dof itself
                                            if (global_index_overrigth == local_dof_indices_2[n]) {

                                                if (this->nsparam.particulesParameters.order==2)
                                                this->system_matrix.set(global_index_overrigth, local_dof_indices_2[n],
                                                                        -1*this->fe.shape_value(n, second_point_v)*sum_line+ 2*sum_line );
                                                if (this->nsparam.particulesParameters.order>2)
                                                this->system_matrix.set(global_index_overrigth, local_dof_indices_2[n],
                                                                       3*this->fe.shape_value(n, second_point_v)*sum_line+ 6*sum_line + -8*this->fe.shape_value(n, third_point_v)*sum_line );
                                            }
                                            // then the third point trough interpolation from the dof of the cell in which the third point is
                                            else {
                                                if (this->nsparam.particulesParameters.order==2)
                                                this->system_matrix.set(global_index_overrigth, local_dof_indices_2[n],
                                                                        -1*this->fe.shape_value(n, second_point_v)*sum_line);
                                                if (this->nsparam.particulesParameters.order>2)
                                                this->system_matrix.set(global_index_overrigth, local_dof_indices_2[n],
                                                                        3*this->fe.shape_value(n, second_point_v)*sum_line + -8*this->fe.shape_value(n, third_point_v)*sum_line );
                                            }
                                        }
                                    }

                                    // define our second point and last to be define the immersed boundary one  this point is where we applied the boundary condition as a dirichlet
                                    if (initial_step) {
                                        // different boundary condition depending if the odf is vx or vy and if the problem we solve
                                        if (k == 0) {
                                            if (dim==2) {
                                                double vx=-particules[p][4]*particules[p][5]*((support_points[local_dof_indices[l]] -
                                                                                               center_immersed) /
                                                                                              (support_points[local_dof_indices[l]] -
                                                                                               center_immersed).norm())[1]+particules[p][2];

                                                this->system_rhs(global_index_overrigth) =vx*sum_line;
                                            }
                                            if (dim==3) {
                                                double vx=particules[p][2];

                                                this->system_rhs(global_index_overrigth) =vx*sum_line;
                                            }

                                        }
                                        else if(k==1){
                                            if (dim==2) {
                                                double vy=particules[p][4]*particules[p][5]*((support_points[local_dof_indices[l]] -
                                                                                               center_immersed) /
                                                                                              (support_points[local_dof_indices[l]] -
                                                                                               center_immersed).norm())[0]+particules[p][3];
                                                this->system_rhs(global_index_overrigth) =vy*sum_line;
                                            }
                                            if (dim==3) {
                                                double vy=particules[p][3];
                                                this->system_rhs(global_index_overrigth) =vy*sum_line;
                                            }
                                        }
                                        else if(k==2 & dim==3){
                                            double vz=particules[p][5];
                                            this->system_rhs(global_index_overrigth) =vz*sum_line;
                                        }
                                    }
                                    else {
                                        this->system_rhs(global_index_overrigth) = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    system_matrix.compress(VectorOperation::insert);
    this->system_rhs.compress(VectorOperation::insert);
    initial_step_bool=false;
}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::set_solution_vector(double value)
{
  this->present_solution = value;
}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::setup_dofs()
{
  TimerOutput::Scope t(this->computing_timer, "setup_dofs");

  // Clear the preconditioner before the matrix they are associated with is
  // cleared
  amg_preconditioner.reset();
  ilu_preconditioner.reset();

  // Now reset system matrix
  system_matrix.clear();

  this->dof_handler.distribute_dofs(this->fe);
  DoFRenumbering::Cuthill_McKee(this->dof_handler);

  this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          this->locally_relevant_dofs);

  const MappingQ<dim>        mapping(this->degreeVelocity_,
                              this->nsparam.femParameters.qmapping_all);
  FEValuesExtractors::Vector velocities(0);

  // Non-zero constraints
  {
    this->nonzero_constraints.clear();

    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            this->nonzero_constraints);
    for (unsigned int i_bc = 0; i_bc < this->nsparam.boundaryConditions.size;
         ++i_bc)
      {
        if (this->nsparam.boundaryConditions.type[i_bc] ==
            BoundaryConditions::BoundaryType::noslip)
          {
            VectorTools::interpolate_boundary_values(
              mapping,
              this->dof_handler,
              this->nsparam.boundaryConditions.id[i_bc],
              ZeroFunction<dim>(dim + 1),
              this->nonzero_constraints,
              this->fe.component_mask(velocities));
          }
        else if (this->nsparam.boundaryConditions.type[i_bc] ==
                 BoundaryConditions::BoundaryType::slip)
          {
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert(
              this->nsparam.boundaryConditions.id[i_bc]);
            VectorTools::compute_no_normal_flux_constraints(
              this->dof_handler,
              0,
              no_normal_flux_boundaries,
              this->nonzero_constraints);
          }
        else if (this->nsparam.boundaryConditions.type[i_bc] ==
                 BoundaryConditions::BoundaryType::function)
          {
            VectorTools::interpolate_boundary_values(
              mapping,
              this->dof_handler,
              this->nsparam.boundaryConditions.id[i_bc],
              FunctionDefined<dim>(
                &this->nsparam.boundaryConditions.bcFunctions[i_bc].u,
                &this->nsparam.boundaryConditions.bcFunctions[i_bc].v,
                &this->nsparam.boundaryConditions.bcFunctions[i_bc].w),
              this->nonzero_constraints,
              this->fe.component_mask(velocities));
          }

        else if (this->nsparam.boundaryConditions.type[i_bc] ==
                 BoundaryConditions::BoundaryType::periodic)
          {
            DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
              this->dof_handler,
              this->nsparam.boundaryConditions.id[i_bc],
              this->nsparam.boundaryConditions.periodic_id[i_bc],
              this->nsparam.boundaryConditions.periodic_direction[i_bc],
              this->nonzero_constraints);
          }
      }
  }
  this->nonzero_constraints.close();

  {
    this->zero_constraints.clear();
    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            this->zero_constraints);

    for (unsigned int i_bc = 0; i_bc < this->nsparam.boundaryConditions.size;
         ++i_bc)
      {
        if (this->nsparam.boundaryConditions.type[i_bc] ==
            BoundaryConditions::BoundaryType::slip)
          {
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert(
              this->nsparam.boundaryConditions.id[i_bc]);
            VectorTools::compute_no_normal_flux_constraints(
              this->dof_handler,
              0,
              no_normal_flux_boundaries,
              this->zero_constraints);
          }
        else if (this->nsparam.boundaryConditions.type[i_bc] ==
                 BoundaryConditions::BoundaryType::periodic)
          {
            DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
              this->dof_handler,
              this->nsparam.boundaryConditions.id[i_bc],
              this->nsparam.boundaryConditions.periodic_id[i_bc],
              this->nsparam.boundaryConditions.periodic_direction[i_bc],
              this->zero_constraints);
          }
        else // if(nsparam.boundaryConditions.boundaries[i_bc].type==Parameters::noslip
          // || Parameters::function)
          {
            VectorTools::interpolate_boundary_values(
              mapping,
              this->dof_handler,
              this->nsparam.boundaryConditions.id[i_bc],
              ZeroFunction<dim>(dim + 1),
              this->zero_constraints,
              this->fe.component_mask(velocities));
          }
      }
  }
  this->zero_constraints.close();

  this->present_solution.reinit(this->locally_owned_dofs,
                                this->locally_relevant_dofs,
                                this->mpi_communicator);
  this->solution_m1.reinit(this->locally_owned_dofs,
                           this->locally_relevant_dofs,
                           this->mpi_communicator);
  this->solution_m2.reinit(this->locally_owned_dofs,
                           this->locally_relevant_dofs,
                           this->mpi_communicator);
  this->solution_m3.reinit(this->locally_owned_dofs,
                           this->locally_relevant_dofs,
                           this->mpi_communicator);

  this->newton_update.reinit(this->locally_owned_dofs, this->mpi_communicator);
  this->system_rhs.reinit(this->locally_owned_dofs, this->mpi_communicator);
  this->local_evaluation_point.reinit(this->locally_owned_dofs,
                                      this->mpi_communicator);

  DynamicSparsityPattern dsp(this->locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  dsp,
                                  this->nonzero_constraints,
                                  false);
  SparsityTools::distribute_sparsity_pattern(
    dsp,
    this->dof_handler.compute_n_locally_owned_dofs_per_processor(),
    this->mpi_communicator,
    this->locally_relevant_dofs);
  system_matrix.reinit(this->locally_owned_dofs,
                       this->locally_owned_dofs,
                       dsp,
                       this->mpi_communicator);

  this->globalVolume_ = GridTools::volume(*this->triangulation);

  this->pcout << "   Number of active cells:       "
              << this->triangulation->n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: "
              << this->dof_handler.n_dofs() << std::endl;
  this->pcout << "   Volume of triangulation:      " << this->globalVolume_
              << std::endl;
}

template <int dim>
template <bool                                              assemble_matrix,
        Parameters::SimulationControl::TimeSteppingMethod scheme>
void
GLSNavierStokesSharpSolver<dim>::assembleGLS()
{
    //std::cout << "this MPI porcess start matrix assemble : "<< this->this_mpi_process<< std::endl;
    MPI_Barrier(this->mpi_communicator);
    if (assemble_matrix)
        system_matrix = 0;
    this->system_rhs = 0;

    double         viscosity_ = this->nsparam.physicalProperties.viscosity;
    Function<dim> *l_forcing_function = this->forcing_function;

    QGauss<dim>                      quadrature_formula(this->degreeQuadrature_);
    const MappingQ<dim>              mapping(this->degreeVelocity_,
                                             this->nsparam.femParameters.qmapping_all);
    FEValues<dim>                    fe_values(mapping,
                                               this->fe,
                                               quadrature_formula,
                                               update_values | update_quadrature_points |
                                               update_JxW_values | update_gradients |
                                               update_hessians);
    const unsigned int               dofs_per_cell = this->fe.dofs_per_cell;
    const unsigned int               n_q_points    = quadrature_formula.size();
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    FullMatrix<double>               local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>                   local_rhs(dofs_per_cell);
    std::vector<Vector<double>> rhs_force(n_q_points, Vector<double>(dim + 1));
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<Tensor<1, dim>>          present_velocity_values(n_q_points);
    std::vector<Tensor<2, dim>>          present_velocity_gradients(n_q_points);
    std::vector<double>                  present_pressure_values(n_q_points);
    std::vector<Tensor<1, dim>>          present_pressure_gradients(n_q_points);
    std::vector<Tensor<1, dim>>          present_velocity_laplacians(n_q_points);
    std::vector<Tensor<2, dim>>          present_velocity_hess(n_q_points);

    Tensor<1, dim> force;

    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<3, dim>> hess_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> laplacian_phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);
    std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

    // Values at previous time step for transient schemes
    std::vector<Tensor<1, dim>> p1_velocity_values(n_q_points);
    std::vector<Tensor<1, dim>> p2_velocity_values(n_q_points);
    std::vector<Tensor<1, dim>> p3_velocity_values(n_q_points);


    // support point
    MappingQ1<dim> immersed_map;
    std::map< types::global_dof_index, Point< dim >>  	support_points;
    DoFTools::map_dofs_to_support_points(immersed_map,this->dof_handler,support_points);

    Point<dim> center_immersed;


    // Time steps and inverse time steps which is used for numerous calculations
    const double dt  = this->simulationControl.getTimeSteps()[0];
    const double sdt = 1. / dt;

    // Vector for the BDF coefficients
    // The coefficients are stored in the following fashion :
    // 0 - n+1
    // 1 - n
    // 2 - n-1
    // 3 - n-2
    Vector<double> bdf_coefs;
    if (scheme == Parameters::SimulationControl::TimeSteppingMethod::bdf1)
        bdf_coefs = bdf_coefficients(1, this->simulationControl.getTimeSteps());

    if (scheme == Parameters::SimulationControl::TimeSteppingMethod::bdf2)
        bdf_coefs = bdf_coefficients(2, this->simulationControl.getTimeSteps());

    if (scheme == Parameters::SimulationControl::TimeSteppingMethod::bdf3)
        bdf_coefs = bdf_coefficients(3, this->simulationControl.getTimeSteps());

    // Matrix of coefficients for the SDIRK methods
    // The lines store the information required for each step
    // Column 0 always refer to outcome of the step that is being calculated
    // Column 1 always refer to step n
    // Column 2+ refer to intermediary steps
    FullMatrix<double> sdirk_coefs;
    if (is_sdirk2(scheme))
        sdirk_coefs = sdirk_coefficients(2, dt);

    if (is_sdirk3(scheme))
        sdirk_coefs = sdirk_coefficients(3, dt);



    // Element size
    double h;

    typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler
            .begin_active(),
            endc = this->dof_handler.end();
    for (; cell != endc; ++cell)
    {
        bool assemble_bool=true;
        if (cell->is_locally_owned()) {

            cell->get_dof_indices(local_dof_indices);
            for (unsigned int k = 0; k < particules.size(); ++k){
                unsigned int count_small = 0;
                if (dim==2) {
                    center_immersed(0)=particules[k][0];
                    center_immersed(1)=particules[k][1];}
                else if (dim==3){
                    center_immersed(0)=particules[k][0];
                    center_immersed(1)=particules[k][1];
                    center_immersed(2)=particules[k][2]; }

                for (unsigned int j = 0; j < local_dof_indices.size(); ++j) {
                    //count the number of dof that are smaller or larger then the radius of the particules
                    //if all the dof are on one side the cell is not cut by the boundary meaning we dont have to do anything
                    if ((support_points[local_dof_indices[j]] - center_immersed).norm() <= particules[k][particules[k].size()-1]) {
                        ++count_small;
                    }
                }
                if(this->nsparam.particulesParameters.assemble_inside and this->nsparam.particulesParameters.P_assemble==Parameters::Particule_Assemble_type::NS){
                    if (count_small != 0 and count_small!= local_dof_indices.size()){
                        assemble_bool = false;
                        break;
                    }
                }
                else {
                    if (count_small != 0) {
                        assemble_bool = false;
                        break;
                    }
                }
            }
            // assemble the cell only if the cell is not cut by a IB

            if (assemble_bool==true ) {



            fe_values.reinit(cell);

            if (dim == 2)
                h = std::sqrt(4. * cell->measure() / M_PI) / this->degreeVelocity_;
            else if (dim == 3)
                h =
                        pow(6 * cell->measure() / M_PI, 1. / 3.) / this->degreeVelocity_;

            local_matrix = 0;
            local_rhs = 0;

            // Gather velocity (values, gradient and laplacian)
            fe_values[velocities].get_function_values(this->evaluation_point,
                                                      present_velocity_values);
            fe_values[velocities].get_function_gradients(
                    this->evaluation_point, present_velocity_gradients);
            fe_values[velocities].get_function_laplacians(
                    this->evaluation_point, present_velocity_laplacians);

            // Gather pressure (values, gradient)
            fe_values[pressure].get_function_values(this->evaluation_point,
                                                    present_pressure_values);
            fe_values[pressure].get_function_gradients(
                    this->evaluation_point, present_pressure_gradients);


            // Calculate forcing term if there is a forcing function
            if (l_forcing_function)
                l_forcing_function->vector_value_list(
                        fe_values.get_quadrature_points(), rhs_force);

            // Gather the previous time steps depending on the number of stages
            // of the time integration scheme
            if (scheme !=
                Parameters::SimulationControl::TimeSteppingMethod::steady)
                fe_values[velocities].get_function_values(this->solution_m1,
                                                          p1_velocity_values);

            if (time_stepping_method_has_two_stages(scheme))
                fe_values[velocities].get_function_values(this->solution_m2,
                                                          p2_velocity_values);

            if (time_stepping_method_has_three_stages(scheme))
                fe_values[velocities].get_function_values(this->solution_m3,
                                                          p3_velocity_values);

            // Loop over the quadrature points
            for (unsigned int q = 0; q < n_q_points; ++q) {
                // Calculation of the magnitude of the velocity for the
                // stabilization parameter
                const double u_mag = std::max(present_velocity_values[q].norm(),
                                              1e-12 * GLS_u_scale);

                // Calculation of the GLS stabilization parameter. The
                // stabilization parameter used is different if the simulation is
                // steady or unsteady. In the unsteady case it includes the value
                // of the time-step
                double tau;
                if (scheme ==
                    Parameters::SimulationControl::TimeSteppingMethod::steady)
                    tau = 1. / std::sqrt(std::pow(2. * u_mag / h, 2) +
                                         9 * std::pow(4 * viscosity_ / (h * h), 2));
                else
                    tau = 1. /
                          std::sqrt(std::pow(sdt, 2) + std::pow(2. * u_mag / h, 2) +
                                    9 * std::pow(4 * viscosity_ / (h * h), 2));

                // Gather the shape functions, their gradient and their laplacian
                // for the velocity and the pressure
                for (unsigned int k = 0; k < dofs_per_cell; ++k) {
                    div_phi_u[k] = fe_values[velocities].divergence(k, q);
                    grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                    phi_u[k] = fe_values[velocities].value(k, q);
                    hess_phi_u[k] = fe_values[velocities].hessian(k, q);
                    phi_p[k] = fe_values[pressure].value(k, q);
                    grad_phi_p[k] = fe_values[pressure].gradient(k, q);

                    for (int d = 0; d < dim; ++d)
                        laplacian_phi_u[k][d] = trace(hess_phi_u[k][d]);
                }

                // Establish the force vector
                for (int i = 0; i < dim; ++i) {
                    const unsigned int component_i =
                            this->fe.system_to_component_index(i).first;
                    force[i] = rhs_force[q](component_i);
                }

                // Calculate the divergence of the velocity
                double present_velocity_divergence =
                        trace(present_velocity_gradients[q]);

                // Calculate the strong residual for GLS stabilization
                auto strong_residual =
                        present_velocity_gradients[q] * present_velocity_values[q] +
                        present_pressure_gradients[q] -
                        viscosity_ * present_velocity_laplacians[q] - force;

                /* Adjust the strong residual in cases where the scheme is
                 transient.
                 The BDF schemes require values at previous time steps which are
                 stored in the p1, p2 and p3 vectors. The SDIRK scheme require the
                 values at the different stages, which are also stored in the same
                 arrays.
                 */

                if (scheme ==
                    Parameters::SimulationControl::TimeSteppingMethod::bdf1)
                    strong_residual += bdf_coefs[0] * present_velocity_values[q] +
                                       bdf_coefs[1] * p1_velocity_values[q];

                if (scheme ==
                    Parameters::SimulationControl::TimeSteppingMethod::bdf2)
                    strong_residual += bdf_coefs[0] * present_velocity_values[q] +
                                       bdf_coefs[1] * p1_velocity_values[q] +
                                       bdf_coefs[2] * p2_velocity_values[q];

                if (scheme ==
                    Parameters::SimulationControl::TimeSteppingMethod::bdf3)
                    strong_residual += bdf_coefs[0] * present_velocity_values[q] +
                                       bdf_coefs[1] * p1_velocity_values[q] +
                                       bdf_coefs[2] * p2_velocity_values[q] +
                                       bdf_coefs[3] * p3_velocity_values[q];


                if (is_sdirk_step1(scheme))
                    strong_residual +=
                            sdirk_coefs[0][0] * present_velocity_values[q] +
                            sdirk_coefs[0][1] * p1_velocity_values[q];

                if (is_sdirk_step2(scheme)) {
                    strong_residual +=
                            sdirk_coefs[1][0] * present_velocity_values[q] +
                            sdirk_coefs[1][1] * p1_velocity_values[q] +
                            sdirk_coefs[1][2] * p2_velocity_values[q];
                }

                if (is_sdirk_step3(scheme)) {
                    strong_residual +=
                            sdirk_coefs[2][0] * present_velocity_values[q] +
                            sdirk_coefs[2][1] * p1_velocity_values[q] +
                            sdirk_coefs[2][2] * p2_velocity_values[q] +
                            sdirk_coefs[2][3] * p3_velocity_values[q];
                }

                // Matrix assembly
                if (assemble_matrix) {
                    // We loop over the column first to prevent recalculation of
                    // the strong jacobian in the inner loop
                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        auto strong_jac =
                                (present_velocity_gradients[q] * phi_u[j] +
                                 grad_phi_u[j] * present_velocity_values[q] +
                                 grad_phi_p[j] - viscosity_ * laplacian_phi_u[j]);

                        if (is_bdf(scheme))
                            strong_jac += phi_u[j] * bdf_coefs[0];
                        if (is_sdirk(scheme))
                            strong_jac += phi_u[j] * sdirk_coefs[0][0];

                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            local_matrix(i, j) +=
                                    (
                                            // Momentum terms
                                            viscosity_ *
                                            scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                                            present_velocity_gradients[q] * phi_u[j] *
                                            phi_u[i] +
                                            grad_phi_u[j] * present_velocity_values[q] *
                                            phi_u[i] -
                                            div_phi_u[i] * phi_p[j] +
                                            // Continuity
                                            phi_p[i] * div_phi_u[j]) *
                                    fe_values.JxW(q);

                            // Mass matrix
                            if (is_bdf(scheme))
                                local_matrix(i, j) += phi_u[j] * phi_u[i] *
                                                      bdf_coefs[0] *
                                                      fe_values.JxW(q);

                            if (is_sdirk(scheme))
                                local_matrix(i, j) += phi_u[j] * phi_u[i] *
                                                      sdirk_coefs[0][0] *
                                                      fe_values.JxW(q);


                            local_matrix(i, j) +=
                                        tau * strong_jac * grad_phi_p[i] * fe_values.JxW(q);


                            // PSPG TAU term is currently disabled because it does
                            // not alter the matrix sufficiently
                            // local_matrix(i, j) +=
                            //  -tau * tau * tau * 4 / h / h *
                            //  (present_velocity_values[q] * phi_u[j]) *
                            //  strong_residual * grad_phi_p[i] *
                            //  fe_values.JxW(q);

                            // Jacobian is currently incomplete
                            if (SUPG) {
                                local_matrix(i, j) +=
                                        tau *
                                        (strong_jac * (grad_phi_u[i] *
                                                       present_velocity_values[q]) +
                                         strong_residual * (grad_phi_u[i] * phi_u[j])) *
                                        fe_values.JxW(q);

                                // SUPG TAU term is currently disabled because it
                                // does not alter the matrix sufficiently
                                // local_matrix(i, j)
                                // +=
                                //   -strong_residual
                                //   * (grad_phi_u[i]
                                //   *
                                //   present_velocity_values[q])
                                //   * tau * tau *
                                //   tau * 4 / h / h
                                //   *
                                //   (present_velocity_values[q]
                                //   * phi_u[j]) *
                                //   fe_values.JxW(q);
                            }
                        }
                    }
                }

                // Assembly of the right-hand side
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    // Navier-Stokes Residual
                    local_rhs(i) +=
                            (
                                    // Momentum
                                    -viscosity_ *
                                    scalar_product(present_velocity_gradients[q],
                                                   grad_phi_u[i]) -
                                    present_velocity_gradients[q] *
                                    present_velocity_values[q] * phi_u[i] +
                                    present_pressure_values[q] * div_phi_u[i] +
                                    force * phi_u[i] -
                                    // Continuity
                                    present_velocity_divergence * phi_p[i]) *
                            fe_values.JxW(q);

                    // Residual associated with BDF schemes
                    if (scheme ==
                        Parameters::SimulationControl::TimeSteppingMethod::bdf1)
                        local_rhs(i) -=
                                bdf_coefs[0] *
                                (present_velocity_values[q] - p1_velocity_values[q]) *
                                phi_u[i] * fe_values.JxW(q);

                    if (scheme ==
                        Parameters::SimulationControl::TimeSteppingMethod::bdf2)
                        local_rhs(i) -=
                                (bdf_coefs[0] * (present_velocity_values[q] * phi_u[i]) +
                                 bdf_coefs[1] * (p1_velocity_values[q] * phi_u[i]) +
                                 bdf_coefs[2] * (p2_velocity_values[q] * phi_u[i])) *
                                fe_values.JxW(q);

                    if (scheme ==
                        Parameters::SimulationControl::TimeSteppingMethod::bdf3)
                        local_rhs(i) -=
                                (bdf_coefs[0] * (present_velocity_values[q] * phi_u[i]) +
                                 bdf_coefs[1] * (p1_velocity_values[q] * phi_u[i]) +
                                 bdf_coefs[2] * (p2_velocity_values[q] * phi_u[i]) +
                                 bdf_coefs[3] * (p3_velocity_values[q] * phi_u[i])) *
                                fe_values.JxW(q);

                    // Residuals associated with SDIRK schemes
                    if (is_sdirk_step1(scheme))
                        local_rhs(i) -=
                                (sdirk_coefs[0][0] *
                                 (present_velocity_values[q] * phi_u[i]) +
                                 sdirk_coefs[0][1] * (p1_velocity_values[q] * phi_u[i])) *
                                fe_values.JxW(q);

                    if (is_sdirk_step2(scheme)) {
                        local_rhs(i) -=
                                (sdirk_coefs[1][0] *
                                 (present_velocity_values[q] * phi_u[i]) +
                                 sdirk_coefs[1][1] *
                                 (p1_velocity_values[q] * phi_u[i]) +
                                 sdirk_coefs[1][2] *
                                 (p2_velocity_values[q] * phi_u[i])) *
                                fe_values.JxW(q);
                    }

                    if (is_sdirk_step3(scheme)) {
                        local_rhs(i) -=
                                (sdirk_coefs[2][0] *
                                 (present_velocity_values[q] * phi_u[i]) +
                                 sdirk_coefs[2][1] *
                                 (p1_velocity_values[q] * phi_u[i]) +
                                 sdirk_coefs[2][2] *
                                 (p2_velocity_values[q] * phi_u[i]) +
                                 sdirk_coefs[2][3] *
                                 (p3_velocity_values[q] * phi_u[i])) *
                                fe_values.JxW(q);
                    }

                    // PSPG GLS term
                    local_rhs(i) +=
                            -tau * (strong_residual * grad_phi_p[i]) * fe_values.JxW(q);

                    // SUPG GLS term
                    if (SUPG) {
                        local_rhs(i) +=
                                -tau *
                                (strong_residual *
                                 (grad_phi_u[i] * present_velocity_values[q])) *
                                fe_values.JxW(q);
                    }
                }
            }

            cell->get_dof_indices(local_dof_indices);

            // The non-linear solver assumes that the nonzero constraints have
            // already been applied to the solution
            const AffineConstraints<double> &constraints_used =
                    this->zero_constraints;
            // initial_step ? nonzero_constraints : zero_constraints;
            if (assemble_matrix) {
                constraints_used.distribute_local_to_global(local_matrix,
                                                            local_rhs,
                                                            local_dof_indices,
                                                            system_matrix,
                                                            this->system_rhs);
            } else {
                constraints_used.distribute_local_to_global(local_rhs,
                                                            local_dof_indices,
                                                            this->system_rhs);
            }
        }
            else if(this->nsparam.particulesParameters.P_assemble==Parameters::Particule_Assemble_type::mass ){
                for (unsigned int q = 0; q < n_q_points; ++q) {
                    if (assemble_matrix) {
                        for (unsigned int i = 0; i< dofs_per_cell; ++i) {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                                local_matrix(i, j) +=(phi_u[i]*phi_u[j]+phi_p[j] * phi_p[i]) *fe_values.JxW(q);
                            }
                            local_rhs(i) =0;

                        }
                    }
                }

                    cell->get_dof_indices(local_dof_indices);

                    // The non-linear solver assumes that the nonzero constraints have
                    // already been applied to the solution
                    const AffineConstraints<double> &constraints_used =
                            this->zero_constraints;
                    // initial_step ? nonzero_constraints : zero_constraints;
                    if (assemble_matrix) {
                        constraints_used.distribute_local_to_global(local_matrix,
                                                                    local_rhs,
                                                                    local_dof_indices,
                                                                    system_matrix,
                                                                    this->system_rhs);
                    } else {
                        constraints_used.distribute_local_to_global(local_rhs,
                                                                    local_dof_indices,
                                                                    this->system_rhs);
                    }

                }
            else{

            }
            }
        }
    //std::cout << "this MPI porcess finish matrix assemble and start compress : "<< this->this_mpi_process<< std::endl;
    if (assemble_matrix)
        system_matrix.compress(VectorOperation::add);
    this->system_rhs.compress(VectorOperation::add);
    //std::cout << "this MPI porcess finish compress : "<< this->this_mpi_process<< std::endl;
}

/**
 * Set the initial condition using a L2 or a viscous solver
 **/
template <int dim>
void
GLSNavierStokesSharpSolver<dim>::set_initial_condition(
  Parameters::InitialConditionType initial_condition_type,
  bool                             restart)
{
  if (restart)
    {
      this->pcout << "************************" << std::endl;
      this->pcout << "---> Simulation Restart " << std::endl;
      this->pcout << "************************" << std::endl;
      this->read_checkpoint();
    }
  else if (initial_condition_type ==
           Parameters::InitialConditionType::L2projection)
    {
      assemble_L2_projection();
      solve_system_GMRES(true, 1e-15, 1e-15, true);
      this->present_solution = this->newton_update;
      this->finish_time_step();
      this->postprocess(true);
    }
  else if (initial_condition_type == Parameters::InitialConditionType::nodal)
    {
      set_nodal_values();
      this->finish_time_step();
      this->postprocess(true);
    }

  else if (initial_condition_type == Parameters::InitialConditionType::viscous)
    {
      set_nodal_values();
      double viscosity = this->nsparam.physicalProperties.viscosity;
      this->nsparam.physicalProperties.viscosity =
        this->nsparam.initialCondition->viscosity;
      Parameters::SimulationControl::TimeSteppingMethod previousControl =
        this->simulationControl.getMethod();
      this->simulationControl.setMethod(
        Parameters::SimulationControl::TimeSteppingMethod::steady);
      PhysicsSolver<TrilinosWrappers::MPI::Vector>::solve_non_linear_system(
        Parameters::SimulationControl::TimeSteppingMethod::steady, false, true);
      this->simulationControl.setMethod(previousControl);
      this->finish_time_step();
      this->postprocess(true);
      this->simulationControl.setMethod(previousControl);
      this->nsparam.physicalProperties.viscosity = viscosity;
    }
  else
    {
      throw std::runtime_error("GLSNS - Initial condition could not be set");
    }
}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::assemble_L2_projection()
{
  system_matrix    = 0;
  this->system_rhs = 0;
  QGauss<dim>                 quadrature_formula(this->degreeQuadrature_);
  const MappingQ<dim>         mapping(this->degreeVelocity_,
                              this->nsparam.femParameters.qmapping_all);
  FEValues<dim>               fe_values(mapping,
                          this->fe,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values);
  const unsigned int          dofs_per_cell = this->fe.dofs_per_cell;
  const unsigned int          n_q_points    = quadrature_formula.size();
  FullMatrix<double>          local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>              local_rhs(dofs_per_cell);
  std::vector<Vector<double>> initial_velocity(n_q_points,
                                               Vector<double>(dim + 1));
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const FEValuesExtractors::Vector     velocities(0);
  const FEValuesExtractors::Scalar     pressure(dim);

  Tensor<1, dim> rhs_initial_velocity_pressure;
  double         rhs_initial_pressure;

  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler
                                                          .begin_active(),
                                                 endc = this->dof_handler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          local_matrix = 0;
          local_rhs    = 0;
          this->nsparam.initialCondition->uvwp.vector_value_list(
            fe_values.get_quadrature_points(), initial_velocity);
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_p[k] = fe_values[pressure].value(k, q);
                  phi_u[k] = fe_values[velocities].value(k, q);
                }

              // Establish the rhs tensor operator
              for (int i = 0; i < dim; ++i)
                {
                  const unsigned int component_i =
                    this->fe.system_to_component_index(i).first;
                  rhs_initial_velocity_pressure[i] =
                    initial_velocity[q](component_i);
                }
              rhs_initial_pressure = initial_velocity[q](dim);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  // Matrix assembly
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      local_matrix(i, j) +=
                        (phi_u[j] * phi_u[i]) * fe_values.JxW(q);
                      local_matrix(i, j) +=
                        (phi_p[j] * phi_p[i]) * fe_values.JxW(q);
                    }
                  local_rhs(i) += (phi_u[i] * rhs_initial_velocity_pressure +
                                   phi_p[i] * rhs_initial_pressure) *
                                  fe_values.JxW(q);
                }
            }

          cell->get_dof_indices(local_dof_indices);
          const AffineConstraints<double> &constraints_used =
            this->nonzero_constraints;
          constraints_used.distribute_local_to_global(local_matrix,
                                                      local_rhs,
                                                      local_dof_indices,
                                                      system_matrix,
                                                      this->system_rhs);
        }
    }
  system_matrix.compress(VectorOperation::add);
  this->system_rhs.compress(VectorOperation::add);
}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::set_nodal_values()
{

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  const MappingQ<dim>              mapping(this->degreeVelocity_,
                              this->nsparam.femParameters.qmapping_all);
  VectorTools::interpolate(mapping,
                           this->dof_handler,
                           this->nsparam.initialCondition->uvwp,
                           this->newton_update,
                           this->fe.component_mask(velocities));
  VectorTools::interpolate(mapping,
                           this->dof_handler,
                           this->nsparam.initialCondition->uvwp,
                           this->newton_update,
                           this->fe.component_mask(pressure));
  this->nonzero_constraints.distribute(this->newton_update);
  this->present_solution = this->newton_update;
}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::assemble_matrix_and_rhs(
  const Parameters::SimulationControl::TimeSteppingMethod time_stepping_method)
{
  TimerOutput::Scope t(this->computing_timer, "assemble_system");


  if (time_stepping_method ==
      Parameters::SimulationControl::TimeSteppingMethod::bdf1)
    assembleGLS<true,
                Parameters::SimulationControl::TimeSteppingMethod::bdf1>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::bdf2)
    assembleGLS<true,
                Parameters::SimulationControl::TimeSteppingMethod::bdf2>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::bdf3)
    assembleGLS<true,
                Parameters::SimulationControl::TimeSteppingMethod::bdf3>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::sdirk2_1)
    assembleGLS<true,
                Parameters::SimulationControl::TimeSteppingMethod::sdirk2_1>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::sdirk2_2)
    assembleGLS<true,
                Parameters::SimulationControl::TimeSteppingMethod::sdirk2_2>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::sdirk3_1)
    assembleGLS<true,
                Parameters::SimulationControl::TimeSteppingMethod::sdirk3_1>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::sdirk3_2)
    assembleGLS<true,
                Parameters::SimulationControl::TimeSteppingMethod::sdirk3_2>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::sdirk3_3)
    assembleGLS<true,
                Parameters::SimulationControl::TimeSteppingMethod::sdirk3_3>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::steady) {
      assembleGLS<true,
              Parameters::SimulationControl::TimeSteppingMethod::steady>();
  }
    vertices_cell_mapping();
    sharp_edge(initial_step_bool);


}
template <int dim>
void
GLSNavierStokesSharpSolver<dim>::assemble_rhs(
  const Parameters::SimulationControl::TimeSteppingMethod time_stepping_method)
{
  //TimerOutput::Scope t(this->computing_timer, "assemble_rhs");

  if (time_stepping_method ==
      Parameters::SimulationControl::TimeSteppingMethod::bdf1)
    assembleGLS<false,
                Parameters::SimulationControl::TimeSteppingMethod::bdf1>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::bdf2)
    assembleGLS<false,
                Parameters::SimulationControl::TimeSteppingMethod::bdf2>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::bdf3)
    assembleGLS<false,
                Parameters::SimulationControl::TimeSteppingMethod::bdf3>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::sdirk2_1)
    assembleGLS<false,
                Parameters::SimulationControl::TimeSteppingMethod::sdirk2_1>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::sdirk2_2)
    assembleGLS<false,
                Parameters::SimulationControl::TimeSteppingMethod::sdirk2_2>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::sdirk3_1)
    assembleGLS<false,
                Parameters::SimulationControl::TimeSteppingMethod::sdirk3_1>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::sdirk3_2)
    assembleGLS<false,
                Parameters::SimulationControl::TimeSteppingMethod::sdirk3_2>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::sdirk3_3)
    assembleGLS<false,
                Parameters::SimulationControl::TimeSteppingMethod::sdirk3_3>();
  else if (time_stepping_method ==
           Parameters::SimulationControl::TimeSteppingMethod::steady) {
      assembleGLS<false,
              Parameters::SimulationControl::TimeSteppingMethod::steady>();
  }
    vertices_cell_mapping();
    sharp_edge(initial_step_bool);
    

}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::solve_linear_system(const bool initial_step,
                                                const bool renewed_matrix)
{

  const double absolute_residual = this->nsparam.linearSolver.minimum_residual;
  const double relative_residual = this->nsparam.linearSolver.relative_residual;
  if (this->nsparam.linearSolver.solver ==
      Parameters::LinearSolver::SolverType::gmres)
    solve_system_GMRES(initial_step,
                       absolute_residual,
                       relative_residual,
                       renewed_matrix);
  else if (this->nsparam.linearSolver.solver ==
           Parameters::LinearSolver::SolverType::bicgstab)
    solve_system_BiCGStab(initial_step,
                          absolute_residual,
                          relative_residual,
                          renewed_matrix);
  else if (this->nsparam.linearSolver.solver ==
           Parameters::LinearSolver::SolverType::amg)
    solve_system_AMG(initial_step,
                     absolute_residual,
                     relative_residual,
                     renewed_matrix);
  else if (this->nsparam.linearSolver.solver ==
           Parameters::LinearSolver::SolverType::direct)
      solve_system_Direct(initial_step,
                       absolute_residual,
                       relative_residual,
                       renewed_matrix);
  else
    throw(std::runtime_error("This solver is not allowed"));
}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::setup_ILU()
{
  TimerOutput::Scope t(this->computing_timer, "setup_ILU");

  const double ilu_fill = this->nsparam.linearSolver.ilu_precond_fill;
  const double ilu_atol = this->nsparam.linearSolver.ilu_precond_atol;
  const double ilu_rtol = this->nsparam.linearSolver.ilu_precond_rtol;
  TrilinosWrappers::PreconditionILU::AdditionalData preconditionerOptions(
    ilu_fill, ilu_atol, ilu_rtol, 0);

  ilu_preconditioner = std::make_shared<TrilinosWrappers::PreconditionILU>();

  ilu_preconditioner->initialize(system_matrix, preconditionerOptions);
}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::setup_AMG()
{
  TimerOutput::Scope t(this->computing_timer, "setup_AMG");

  std::vector<std::vector<bool>> constant_modes;
  // Constant modes include pressure since everything is in the same matrix
  std::vector<bool> velocity_components(dim + 1, true);
  velocity_components[dim] = true;
  DoFTools::extract_constant_modes(this->dof_handler,
                                   velocity_components,
                                   constant_modes);

  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.constant_modes = constant_modes;

  const bool elliptic              = false;
  bool       higher_order_elements = false;
  if (this->degreeVelocity_ > 1)
    higher_order_elements = true;
  const unsigned int n_cycles = this->nsparam.linearSolver.amg_n_cycles;
  const bool         w_cycle  = this->nsparam.linearSolver.amg_w_cycles;
  const double       aggregation_threshold =
    this->nsparam.linearSolver.amg_aggregation_threshold;
  const unsigned int smoother_sweeps =
    this->nsparam.linearSolver.amg_smoother_sweeps;
  const unsigned int smoother_overlap =
    this->nsparam.linearSolver.amg_smoother_overlap;
  const bool                                        output_details = false;
  const char *                                      smoother_type  = "ILU";
  const char *                                      coarse_type    = "ILU";
  TrilinosWrappers::PreconditionAMG::AdditionalData preconditionerOptions(
    elliptic,
    higher_order_elements,
    n_cycles,
    w_cycle,
    aggregation_threshold,
    constant_modes,
    smoother_sweeps,
    smoother_overlap,
    output_details,
    smoother_type,
    coarse_type);

  Teuchos::ParameterList              parameter_ml;
  std::unique_ptr<Epetra_MultiVector> distributed_constant_modes;
  preconditionerOptions.set_parameters(parameter_ml,
                                       distributed_constant_modes,
                                       system_matrix);
  const double ilu_fill = this->nsparam.linearSolver.amg_precond_ilu_fill;
  const double ilu_atol = this->nsparam.linearSolver.amg_precond_ilu_atol;
  const double ilu_rtol = this->nsparam.linearSolver.amg_precond_ilu_rtol;
  parameter_ml.set("smoother: ifpack level-of-fill", ilu_fill);
  parameter_ml.set("smoother: ifpack absolute threshold", ilu_atol);
  parameter_ml.set("smoother: ifpack relative threshold", ilu_rtol);

  parameter_ml.set("coarse: ifpack level-of-fill", ilu_fill);
  parameter_ml.set("coarse: ifpack absolute threshold", ilu_atol);
  parameter_ml.set("coarse: ifpack relative threshold", ilu_rtol);
  amg_preconditioner = std::make_shared<TrilinosWrappers::PreconditionAMG>();
  amg_preconditioner->initialize(system_matrix, parameter_ml);
}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::solve_system_GMRES(const bool   initial_step,
                                               const double absolute_residual,
                                               const double relative_residual,
                                               const bool   renewed_matrix)
{
  const AffineConstraints<double> &constraints_used =
    initial_step ? this->nonzero_constraints : this->zero_constraints;
  const double linear_solver_tolerance =
    std::max(relative_residual * this->system_rhs.l2_norm(), absolute_residual);

  if (this->nsparam.linearSolver.verbosity != Parameters::Verbosity::quiet)
    {
      this->pcout << "  -Tolerance of iterative solver is : "
                  << std::setprecision(
                       this->nsparam.linearSolver.residual_precision)
                  << linear_solver_tolerance << std::endl;
    }
  TrilinosWrappers::MPI::Vector completely_distributed_solution(
    this->locally_owned_dofs, this->mpi_communicator);

  SolverControl solver_control(this->nsparam.linearSolver.max_iterations,
                               linear_solver_tolerance,
                               true,
                               true);
  solver_control.log_frequency(1);

  TrilinosWrappers::SolverGMRES solver(solver_control);

  if (renewed_matrix || !ilu_preconditioner)
    setup_ILU();

  {
    TimerOutput::Scope t(this->computing_timer, "solve_linear_system");

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 this->system_rhs,
                 *ilu_preconditioner);

    if (this->nsparam.linearSolver.verbosity != Parameters::Verbosity::quiet)
      {
        this->pcout << "  -Iterative solver took : "
                    << solver_control.last_step() << " steps " << std::endl;
      }
  }
  constraints_used.distribute(completely_distributed_solution);
  this->newton_update = completely_distributed_solution;
}


template <int dim>
void
GLSNavierStokesSharpSolver<dim>::solve_system_Direct(const bool   initial_step,
                                                    const double absolute_residual,
                                                    const double relative_residual,
                                                    const bool   renewed_matrix)
{
    const AffineConstraints<double> &constraints_used =
            initial_step ? this->nonzero_constraints : this->zero_constraints;
    const double linear_solver_tolerance =
            std::max(relative_residual * this->system_rhs.l2_norm(), absolute_residual);

    TrilinosWrappers::MPI::Vector completely_distributed_solution(
            this->locally_owned_dofs, this->mpi_communicator);

    SolverControl solver_control(this->nsparam.linearSolver.max_iterations,
                                 linear_solver_tolerance,
                                 true,
                                 true);
    TrilinosWrappers::SolverDirect solver(solver_control);

    if (renewed_matrix || !ilu_preconditioner)
        setup_ILU();
    solver.initialize(system_matrix);
    solver.solve(completely_distributed_solution,this->system_rhs);
    constraints_used.distribute(completely_distributed_solution);
    this->newton_update = completely_distributed_solution;
}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::solve_system_BiCGStab(
  const bool   initial_step,
  const double absolute_residual,
  const double relative_residual,
  const bool   renewed_matrix)
{
  TimerOutput::Scope t(this->computing_timer, "solve");

  const AffineConstraints<double> &constraints_used =
    initial_step ? this->nonzero_constraints : this->zero_constraints;
  const double linear_solver_tolerance =
    std::max(relative_residual * this->system_rhs.l2_norm(), absolute_residual);
  if (this->nsparam.linearSolver.verbosity != Parameters::Verbosity::quiet)
    {
      this->pcout << "  -Tolerance of iterative solver is : "
                  << std::setprecision(
                       this->nsparam.linearSolver.residual_precision)
                  << linear_solver_tolerance << std::endl;
    }
  TrilinosWrappers::MPI::Vector completely_distributed_solution(
    this->locally_owned_dofs, this->mpi_communicator);

  SolverControl solver_control(this->nsparam.linearSolver.max_iterations,
                               linear_solver_tolerance,
                               true,
                               true);
  TrilinosWrappers::SolverBicgstab solver(solver_control);

  if (renewed_matrix || !ilu_preconditioner)
    setup_ILU();

  {
    TimerOutput::Scope t(this->computing_timer, "solve_linear_system");

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 this->system_rhs,
                 *ilu_preconditioner);

    if (this->nsparam.linearSolver.verbosity != Parameters::Verbosity::quiet)
      {
        this->pcout << "  -Iterative solver took : "
                    << solver_control.last_step() << " steps " << std::endl;
      }
    constraints_used.distribute(completely_distributed_solution);
    this->newton_update = completely_distributed_solution;
  }
}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::solve_system_AMG(const bool   initial_step,
                                             const double absolute_residual,
                                             const double relative_residual,
                                             const bool   renewed_matrix)
{
  const AffineConstraints<double> &constraints_used =
    initial_step ? this->nonzero_constraints : this->zero_constraints;

  const double linear_solver_tolerance =
    std::max(relative_residual * this->system_rhs.l2_norm(), absolute_residual);
  if (this->nsparam.linearSolver.verbosity != Parameters::Verbosity::quiet)
    {
      this->pcout << "  -Tolerance of iterative solver is : "
                  << std::setprecision(
                       this->nsparam.linearSolver.residual_precision)
                  << linear_solver_tolerance << std::endl;
    }
  TrilinosWrappers::MPI::Vector completely_distributed_solution(
    this->locally_owned_dofs, this->mpi_communicator);

  SolverControl solver_control(this->nsparam.linearSolver.max_iterations,
                               linear_solver_tolerance,
                               true,
                               true);
  TrilinosWrappers::SolverGMRES solver(solver_control);

  if (renewed_matrix || !amg_preconditioner)
    setup_AMG();

  {
    TimerOutput::Scope t(this->computing_timer, "solve_linear_system");

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 this->system_rhs,
                 *amg_preconditioner);

    if (this->nsparam.linearSolver.verbosity != Parameters::Verbosity::quiet)
      {
        this->pcout << "  -Iterative solver took : "
                    << solver_control.last_step() << " steps " << std::endl;
      }

    constraints_used.distribute(completely_distributed_solution);

    this->newton_update = completely_distributed_solution;
  }
}

template <int dim>
void
GLSNavierStokesSharpSolver<dim>::solve()
{
  this->read_mesh();
  this->create_manifolds();
  this->define_particules();


  this->setup_dofs();
  this->set_initial_condition(this->nsparam.initialCondition->type,
                              this->nsparam.restartParameters.restart);

  initial_step_bool=true;
  while (this->simulationControl.integrate())
    {
      printTime(this->pcout, this->simulationControl);

      if (!this->simulationControl.firstIter())
        {
            refine_ib();
            NavierStokesBase<dim, TrilinosWrappers::MPI::Vector, IndexSet>::
            refine_mesh();

            if (this->simulationControl.getParameters().method == Parameters::SimulationControl::TimeSteppingMethod::steady){
                this->set_initial_condition(this->nsparam.initialCondition->type,
                                            this->nsparam.restartParameters.restart);
                initial_step_bool=true;

            }
        }

      this->iterate(this->simulationControl.firstIter());
//   this->iterate(true);
      this->postprocess(false);
      this->finish_time_step();
      force_on_ib();
    }
    //force_on_ib();
  this->finish_simulation();
}


// Pre-compile the 2D and 3D Navier-Stokes solver to ensure that the library is
// valid before we actually compile the solver This greatly helps with debugging
template class GLSNavierStokesSharpSolver<2>;
template class GLSNavierStokesSharpSolver<3>;
