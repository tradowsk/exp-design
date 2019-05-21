#include "../catalycity/catalycity_runner.h"
#include "../catalycity/model_map.h"

#include "grins/variable_warehouse.h"

// libMesh
#include "libmesh/getpot.h"

#include <queso/Environment.h>
#include <queso/VectorSpace.h>
#include <queso/GslVector.h>
#include <queso/GslMatrix.h>
#include <queso/BoxSubset.h>
#include <queso/UniformVectorRV.h>
#include <queso/GaussianLikelihoodDiagonalCovariance.h>
#include <queso/GenericVectorRV.h>
#include <queso/LinearLagrangeInterpolationSurrogate.h>
#include <queso/InterpolationSurrogateBuilder.h>
#include <queso/InterpolationSurrogateIOASCII.h>
#include <queso/MpiComm.h>

#include <fstream>

template<typename V = QUESO::GslVector, typename M = QUESO::GslMatrix>
class MyInterpolationBuilder : public QUESO::InterpolationSurrogateBuilder<V,M>
{
  public:
    MyInterpolationBuilder( QUESO::InterpolationSurrogateDataSet<V,M> & data, std::string & input_filename, std::string & cat_model, MPI_Comm comm )
      : QUESO::InterpolationSurrogateBuilder<V,M>(data),
        _cat_model(cat_model)
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
    };

    virtual ~MyInterpolationBuilder(){};

    virtual void evaluate_model( const V & domainVector, std::vector<double> & values )
    {
      double p  = domainVector[0];
      double Xo = domainVector[1];
      double Tf = domainVector[2];
    
      double T0 = 298.0;
      double T1 = 0.0750*(Tf-873.0) + 320.0;
      double T3 = 0.1375*(Tf-873.0) + 470.0;
      double T4 = 0.1250*(Tf-873.0) + 350.0;

      std::cout <<"***********************" <<std::endl
                <<"p: " <<p <<std::endl
                <<"Xo: " <<Xo <<std::endl
                <<"Tf: " <<Tf <<std::endl;
    
      for (unsigned int i=3; i<domainVector.sizeGlobal(); ++i)
        {
          std::cout <<"Param " <<i-3 <<": " <<domainVector[i] <<std::endl;
        }

      std::cout <<"***********************" <<std::endl;

      // TODO this is ugly
      GRINS::GRINSPrivate::VariableWarehouse::clear();
      
      GetPot & getpot = const_cast<GetPot &>(_runner->get_input_file());
      
      getpot.set("Materials/TestMaterial/ThermodynamicPressure/value",p);
      getpot.set("BoundaryConditions/Inlet/SpeciesMassFractions/X_O2",1.0-Xo);
      getpot.set("BoundaryConditions/Inlet/SpeciesMassFractions/X_O",Xo);
      
      std::stringstream ss;
      ss <<"(y<0.244)*(((" <<T1 <<"-" <<T0 <<")/(0.244-0.000))*(y-0.000)+" <<T0 <<")+(y>=0.244)*(y<0.324)*(((" <<Tf <<"-" <<T1 <<")/(0.324-0.244))*(y-0.244)+" <<T1 <<")+(y>=0.324)*(y<0.639)*(" <<Tf <<")+(y>=0.639)*(y<0.709)*(((" <<T3 <<"-" <<Tf <<")/(0.709-0.639))*(y-0.639)+" <<Tf <<")+(y>=0.709)*(((" <<T4 <<"-" <<T3 <<")/(0.869-0.709))*(y-0.709)+" <<T3 <<")";
      
      getpot.set("BoundaryConditions/Wall/Temperature/T",ss.str());
      getpot.set("BoundaryConditions/Cylinder/Temperature/T",Tf);
      
      switch(model_map[_cat_model])
        {
          case 0: // constant
            getpot.set("BoundaryConditions/Cylinder/SpeciesMassFractions/ConstantCatalycity/gamma",domainVector[3]);
            break;
          case 3: // reduced_arrhenius
            getpot.set("BoundaryConditions/Cylinder/SpeciesMassFractions/ArrheniusCatalycity/gamma0",domainVector[3]);
            break;
          case 4: // reduced_pwr
            getpot.set("BoundaryConditions/Cylinder/SpeciesMassFractions/PowerLawCatalycity/gamma0",domainVector[3]);
            getpot.set("BoundaryConditions/Cylinder/SpeciesMassFractions/PowerLawCatalycity/alpha",domainVector[4]);
            break;
          default:
            libmesh_error_msg("Unsupported cat model: "+_cat_model);
            break;
        }
      
      _runner->init();
      
      _runner->run();
        
      values[0] =  _runner->get_simulation().get_qoi_value(0);
      
    }
    
  private:
    std::shared_ptr<CatalycityRunner> _runner;
    std::string _cat_model;

};



int main(int argc, char ** argv)
{
  if (argc != 3)
  {
    std::cerr <<"exactly 2 arguments required ./generate_surrogate dim cat_model, found " <<(argc-1) <<std::endl;
    return -1;
  }
  
  MPI_Init(&argc, &argv);
  
    std::string dim = argv[1];
    std::string cat_model = argv[2];

    unsigned int n_params = 3;
    std::vector<double> min;
    std::vector<double> max;
    std::vector<unsigned int> n_points;
    
    // pressure
    min.push_back(200.0);
    max.push_back(350.0);
    n_points.push_back(4);
    
    // Xo
    min.push_back(0.005);
    max.push_back(0.02);
    n_points.push_back(4);
    
    // T
    min.push_back(600);
    max.push_back(1200.0);
    n_points.push_back(4);
    
    
    switch(model_map[cat_model])
      {
        case 0:
          n_params += 1;
          min.push_back(0.010);
          max.push_back(0.200);
          n_points.push_back(101);
          break;
        case 1:
          n_params += 2;
          min.push_back(0.01);
          min.push_back(-1400.0);
          max.push_back(0.10);
          max.push_back(-700.0);
          n_points.push_back(37);
          n_points.push_back(36);
          break;
        case 2:
          n_params += 3;
          min.push_back(0.01);
          min.push_back(700.0);
          min.push_back(-1.5);
          max.push_back(0.10);
          max.push_back(1400.0);
          max.push_back(-0.1);
          n_points.push_back(10);
          n_points.push_back(36);
          n_points.push_back(15);
          break;
        case 3:
          n_params += 1;
          min.push_back(0.01);
          max.push_back(1.0);
          n_points.push_back(100);
          break;
        case 4:
          n_params += 2;
          min.push_back(0.0);
          min.push_back(-2.5);
          max.push_back(1.0);
          max.push_back(-0.1);
          n_points.push_back(10);
          n_points.push_back(25);
          break;
        default:
          std::cerr <<"********* Unrecognized catalycity model: " <<cat_model <<std::endl;
          return -1;
      }
    
    unsigned int n_data = 1;
    
    std::string input_filename = "./"+dim+"d/"+cat_model+"/800.in";
    std::string data_filename = "./"+dim+"d/"+cat_model+"/surrogate_data/surr_pTXo.dat";

    QUESO::FullEnvironment env(MPI_COMM_WORLD,"./"+cat_model+"_surrogate.in", "", NULL);

    QUESO::VectorSpace<> param_space( env, "param_", n_params, NULL );

    QUESO::VectorSpace<> data_space( env, "data_", n_data, NULL );

    QUESO::GslVector min_vals( param_space.zeroVector() );
    for (unsigned int i=0; i<n_params; ++i)
      min_vals[i] = min[i];
    
    QUESO::GslVector max_vals( param_space.zeroVector() );
    for (unsigned int i=0; i<n_params; ++i)
      max_vals[i] = max[i];
      
    QUESO::BoxSubset<> param_domain( "param_domain_", param_space, min_vals, max_vals );
   
    QUESO::InterpolationSurrogateDataSet<QUESO::GslVector, QUESO::GslMatrix> data(param_domain,n_points,n_data);
    
    MyInterpolationBuilder<QUESO::GslVector,QUESO::GslMatrix> builder( data, input_filename, cat_model, env.subComm().Comm() );
    builder.build_values();

    QUESO::InterpolationSurrogateIOASCII<QUESO::GslVector, QUESO::GslMatrix> data_writer;
    data_writer.write(data_filename, data.get_dataset(0));

  MPI_Finalize();

  return 0;
}

