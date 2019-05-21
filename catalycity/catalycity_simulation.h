
#ifndef CATALYCITY_SIMULATION_H
#define CATALYCITY_SIMULATION_H

#include "grins/simulation.h"

class GetPot;

class CatalycitySimulation : public GRINS::Simulation
{
  public:
  
    CatalycitySimulation( const GetPot & input, GetPot & command_line, GRINS::SimulationBuilder & sim_builder, const libMesh::Parallel::Communicator & comm);

    void get_catalycity_params( std::vector<libMesh::Real> & params);

    void set_catalycity_params( std::vector<libMesh::Real> & params);

  private:
    std::vector<libMesh::dof_id_type> _catalytic_bcs;

};

#endif
