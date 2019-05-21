#include "../catalycity/catalycity_runner.h"
#include "../catalycity/model_map.h"

#include "grins/variable_warehouse.h"

#include <queso/GslVector.h>
#include <queso/GslMatrix.h>
#include <queso/LinearLagrangeInterpolationSurrogate.h>
#include <queso/InterpolationSurrogateBuilder.h>
#include <queso/InterpolationSurrogateIOASCII.h>

class GRINSEvaluator {

  public:
    GRINSEvaluator( const std::string & input_filename,
                    const std::string & cat_model,
                    std::shared_ptr<QUESO::LinearLagrangeInterpolationSurrogate<QUESO::GslVector,QUESO::GslMatrix>> & surrogate,
                    std::shared_ptr<QUESO::InterpolationSurrogateIOASCII<QUESO::GslVector,QUESO::GslMatrix>> & surrogate_io,
                    MPI_Comm comm)
    : _cat_model(cat_model),
      _surrogate(surrogate),
      _surrogate_io(surrogate_io)
    {
      char * arv[] = {
                      (char*)"",
                      (char*)input_filename.c_str(),
                      (char*)"-ksp_type",
                      (char*)"preonly",
                      (char*)"-pc_type",
                      (char*)"lu",
                      (char*)"-pc_factor_mat_solver_package",
                      (char*)"mumps"
                     };

      _runner.reset(new CatalycityRunner(8,arv,comm));
    }
    
    double calc_absorption_grins(std::vector<double> cat_params, double p, double Xo, double Tf)
    {
      GRINS::GRINSPrivate::VariableWarehouse::clear();
      
      GetPot & getpot = const_cast<GetPot &>(_runner->get_input_file());
      
      getpot.set("Materials/TestMaterial/ThermodynamicPressure/value",p);
      getpot.set("BoundaryConditions/Inlet/SpeciesMassFractions/X_O2",1.0-Xo);
      getpot.set("BoundaryConditions/Inlet/SpeciesMassFractions/X_O",Xo);
      
      double T0 = 298.0;
      double T1 = 0.0750*(Tf-873.0) + 320.0;
      double T3 = 0.1375*(Tf-873.0) + 470.0;
      double T4 = 0.1250*(Tf-873.0) + 350.0;
      
//      getpot.set("TemperatureProfile/T1",T1);
//      getpot.set("TemperatureProfile/Tf",Tf);
//      getpot.set("TemperatureProfile/T3",T3);
//      getpot.set("TemperatureProfile/T4",T4);
      
      std::stringstream ss;
      ss <<"(y<0.244)*(((" <<T1 <<"-" <<T0 <<")/(0.244-0.000))*(y-0.000)+" <<T0 <<")+(y>=0.244)*(y<0.324)*(((" <<Tf <<"-" <<T1 <<")/(0.324-0.244))*(y-0.244)+" <<T1 <<")+(y>=0.324)*(y<0.639)*(" <<Tf <<")+(y>=0.639)*(y<0.709)*(((" <<T3 <<"-" <<Tf <<")/(0.709-0.639))*(y-0.639)+" <<Tf <<")+(y>=0.709)*(((" <<T4 <<"-" <<T3 <<")/(0.869-0.709))*(y-0.709)+" <<T3 <<")";
      
//      std::cout <<"\n\n\n************ Tprofile: " <<ss.str() <<"\n\n\n";
      
      getpot.set("BoundaryConditions/Wall/Temperature/T",ss.str());
      getpot.set("BoundaryConditions/Cylinder/Temperature/T",Tf);
      
       switch(model_map[_cat_model])
        {
          case 0: // constant
            getpot.set("BoundaryConditions/Cylinder/SpeciesMassFractions/ConstantCatalycity/gamma",cat_params[0]);
            break;
          case 3: // reduced_arrhenius
            getpot.set("BoundaryConditions/Cylinder/SpeciesMassFractions/ArrheniusCatalycity/gamma0",cat_params[0]);
            break;
          case 4: // reduced_pwr
            getpot.set("BoundaryConditions/Cylinder/SpeciesMassFractions/PowerLawCatalycity/gamma0",cat_params[0]);
            getpot.set("BoundaryConditions/Cylinder/SpeciesMassFractions/PowerLawCatalycity/alpha",cat_params[1]);
            break;
          default:
            libmesh_error_msg("Unsupported cat model: "+_cat_model);
            break;
        }
      
      _runner->init();
      
      _runner->run();
        
      return _runner->get_simulation().get_qoi_value(0);
    }
    
    double calc_absorption_surrogate(std::vector<double> cat_params, double p, double Xo, double Tf, const QUESO::BaseEnvironment & env)
    {
      QUESO::VectorSpace<QUESO::GslVector,QUESO::GslMatrix> grins_space( env, "grins_", 3+cat_params.size(), NULL );

      QUESO::GslVector scenario_params( grins_space.zeroVector() );
      scenario_params[0] = p;
      scenario_params[1] = Xo;
      scenario_params[2] = Tf;
      
      for (unsigned int c=0; c<cat_params.size(); ++c)
        scenario_params[3+c] = cat_params[c];
      
      return _surrogate->evaluate(scenario_params);
    }
    
  private:
    std::string _cat_model;
    std::shared_ptr<CatalycityRunner> _runner;
    
    std::shared_ptr<QUESO::LinearLagrangeInterpolationSurrogate<QUESO::GslVector,QUESO::GslMatrix>>  _surrogate;
    std::shared_ptr<QUESO::InterpolationSurrogateIOASCII<QUESO::GslVector,QUESO::GslMatrix>> _surrogate_io;


};

