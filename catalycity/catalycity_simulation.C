
#include "catalycity_simulation.h"

// GRINS
#include "grins/simulation.h"
#include "grins/gas_solid_catalytic_wall.h"
#include "grins/antioch_chemistry.h"

  CatalycitySimulation::CatalycitySimulation( const GetPot & input, GetPot & command_line, GRINS::SimulationBuilder & sim_builder, const libMesh::Parallel::Communicator & comm)
  : GRINS::Simulation(input,command_line,sim_builder,comm)
  {
    unsigned int num_catalytic_bcs = input.vector_variable_size("QUESO/catalytic_bc_ids");
    
    libmesh_assert(num_catalytic_bcs > 0);
    
    for (unsigned int i=0; i<num_catalytic_bcs; ++i)
      {
        libMesh::dof_id_type id = input("QUESO/catalytic_bc_ids",0,i);
        libmesh_assert(id > 0);
        _catalytic_bcs.push_back( id );
      }
  }

  void CatalycitySimulation::get_catalycity_params( std::vector<libMesh::Real> & params)
  {
    std::vector<std::shared_ptr<GRINS::NeumannBCContainer> > & neumann_bcs = this->_multiphysics_system->get_neumann_bcs();
    
    for (unsigned int id_idx=0; id_idx<_catalytic_bcs.size(); ++id_idx)
      {
        for (unsigned int bc=0; bc<neumann_bcs.size(); ++bc)
          {
            if (neumann_bcs[bc]->has_bc_id(_catalytic_bcs[id_idx]))
              {
                std::shared_ptr<GRINS::NeumannBCAbstract> bc_base = neumann_bcs[bc]->get_func();

                GRINS::GasSolidCatalyticWall<GRINS::AntiochChemistry> * wall = libMesh::cast_ptr<GRINS::GasSolidCatalyticWall<GRINS::AntiochChemistry>*>( bc_base.get() );
                
                wall->get_catalycity_params(params);
              }
          }   
      }
  }

  void CatalycitySimulation::set_catalycity_params( std::vector<libMesh::Real> & params)
  {
    std::vector<std::shared_ptr<GRINS::NeumannBCContainer> > & neumann_bcs = this->_multiphysics_system->get_neumann_bcs();
    
    for (unsigned int id_idx=0; id_idx<_catalytic_bcs.size(); ++id_idx)
      {
        for (unsigned int bc=0; bc<neumann_bcs.size(); ++bc)
          {
            if (neumann_bcs[bc]->has_bc_id(_catalytic_bcs[id_idx]))
              {
                std::shared_ptr<GRINS::NeumannBCAbstract> bc_base = neumann_bcs[bc]->get_func();

                GRINS::GasSolidCatalyticWall<GRINS::AntiochChemistry> * wall = libMesh::cast_ptr<GRINS::GasSolidCatalyticWall<GRINS::AntiochChemistry>*>( bc_base.get() );

                wall->set_catalycity_params( params );
                
                std::vector<libMesh::Real> set_params(params.size());
                wall->get_catalycity_params(set_params);
                
//                std::cout <<"***************************" <<std::endl;
//                for (unsigned int p=0; p<set_params.size(); ++p)
//                  std::cout <<"Catalycity param " <<p <<" set to: " <<set_params[p] <<std::endl;
//                std::cout <<"***************************" <<std::endl;
              }
          }   
      }
  }

