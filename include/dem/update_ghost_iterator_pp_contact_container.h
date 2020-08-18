/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - 2020 by the Lethe authors
 *
 * This file is part of the Lethe library
 *
 * The Lethe library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the Lethe distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Shahab Golshan, Polytechnique Montreal, 2019
 */

using namespace dealii;

#ifndef UPDATEGHOSTITERATORPPCONTACTCONTAINER_H_
#define UPDATEGHOSTITERATORPPCONTACTCONTAINER_H_

/**
 * Updates the iterators to particles in local_ghost adjacent_particles
 * (output of pp fine search)
 *
 * @param ghost_adjacent_particles Output of particle-particle fine search
 * @param particle_container Output of update_particle_container function
 */

template <int dim>
void
update_ghost_iterator_pp_contact_container(
  std::map<int, std::map<int, pp_contact_info_struct<dim>>>
    &ghost_adjacent_particles,
  const std::unordered_map<int, Particles::ParticleIterator<dim>>
    &ghost_particle_container)
{
  for (auto adjacent_particles_iterator = ghost_adjacent_particles.begin();
       adjacent_particles_iterator != ghost_adjacent_particles.end();
       ++adjacent_particles_iterator)
    {
      auto pairs_in_contant_content = &adjacent_particles_iterator->second;
      for (auto pp_map_iterator = pairs_in_contant_content->begin();
           pp_map_iterator != pairs_in_contant_content->end();
           ++pp_map_iterator)
        {
          int particle_two_id = pp_map_iterator->first;
          pp_map_iterator->second.particle_two =
            ghost_particle_container.at(particle_two_id);
        }
    }
}

#endif /* UPDATEGHOSTITERATORPPCONTACTCONTAINER_H_ */
