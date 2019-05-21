
#ifndef CATALYCITY_RUNNER_H
#define CATALYCITY_RUNNER_H

#include "grins/runner.h"

#include "libmesh/libmesh.h"

#include "catalycity_simulation.h"

class CatalycityRunner : public GRINS::Runner
{
  public:
    CatalycityRunner(int argc, char* argv[], MPI_Comm comm)
    : GRINS::Runner(argc,argv),
      _libmesh_init(argc,argv,comm) {}
//      _comm(comm) {}
    
    void init()
    {
      // Initialize Simulation
      _simulation.reset( new CatalycitySimulation( *_inputfile,
                                                    _command_line,
                                                    _sim_builder,
                                                    _libmesh_init.comm() ) );
    }
    
    CatalycitySimulation* get_cat_simulation()
    {
      return libMesh::cast_ptr<CatalycitySimulation*>(_simulation.get());
    }

  private:
    libMesh::LibMeshInit _libmesh_init;
//    MPI_Comm _comm;
};

#endif
