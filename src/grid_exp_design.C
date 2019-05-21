#include "../catalycity/grins_evaluator.h"

// libMesh
#include "libmesh/getpot.h"

#include <queso/Environment.h>
#include <queso/VectorSpace.h>
#include <queso/VectorRV.h>
#include <queso/GslVector.h>
#include <queso/GslMatrix.h>
#include <queso/BoxSubset.h>
#include <queso/ScalarFunction.h>
#include <queso/GridSearchExperimentalDesign.h>
#include <queso/ExperimentalLikelihoodInterface.h>
#include <queso/ExperimentalLikelihoodWrapper.h>
#include <queso/GaussianLikelihoodDiagonalCovariance.h>
#include <queso/ExperimentMetricEIG.h>
#include <queso/ExperimentMetricMinVariance.h>
#include <queso/GaussianVectorRV.h>

#include <fstream>

template<typename V = QUESO::GslVector, typename M = QUESO::GslMatrix>
class Likelihood :  public QUESO::GaussianLikelihoodDiagonalCovariance<V,M>
{
  public:

    Likelihood( const QUESO::VectorSet<V,M> & domain,
                const V & observations,
                const V & covariance,
                std::string & cat_model,
                std::shared_ptr<GRINSEvaluator> & grins_eval,
                const std::vector<std::vector<double>> initial_scenarios,
                QUESO::GaussianVectorRV<V,M> & noise)
    : QUESO::GaussianLikelihoodDiagonalCovariance<V,M>("exp_like_",domain,observations,covariance),
      m_fix_p(false),
      m_fix_Xo(false),
      m_fix_Tf(false),
      m_cat_model(cat_model),
      m_grins_eval(grins_eval),
      m_initial_scenarios(initial_scenarios),
      m_noise(noise)
    {}

    void fix_p(double p)
    {
      m_fix_p = true;
      m_p = p;
    }

    void fix_Xo(double Xo)
    {
      m_fix_Xo = true;
      m_Xo = Xo;
    }

    void fix_Tf(double Tf)
    {
      m_fix_Tf = true;
      m_Tf = Tf;
    }

    virtual void evaluateModel(const V & domainVector, V & modelOutput) const override
    {
      unsigned int num_data = this->m_observations.sizeGlobal() - 1;
    
      std::vector<double> cat_params;
      for (unsigned int c=0; c< domainVector.sizeGlobal(); ++c)
        cat_params.push_back(domainVector[c]);
  
      for (unsigned int i = 0; i < num_data; ++i)
      {
        double p  = m_initial_scenarios[i][0];
        double Xo = m_initial_scenarios[i][1];
        double Tf = m_initial_scenarios[i][2];
        
        modelOutput[i] = m_grins_eval->calc_absorption_surrogate(cat_params,p,Xo,Tf,this->m_env);
      }

      modelOutput[num_data] = m_grins_eval->calc_absorption_surrogate(cat_params,m_p,m_Xo,m_Tf,this->m_env);

    }

    virtual double lnValue( const V & domainVector,
                            const V * domainDirection,
                            V * gradVector,
                            M * hessianMatrix,
                            V * hessianEffect) const override
    {
      return QUESO::GaussianLikelihoodDiagonalCovariance<V,M>::lnValue(domainVector);
    }

    void setScenario(std::vector<double> & scenario_params)
    {
      unsigned int index = 0;
      m_p  = m_fix_p  ? m_p  : scenario_params[index++];
      m_Xo = m_fix_Xo ? m_Xo : scenario_params[index++];
      m_Tf = m_fix_Tf ? m_Tf : scenario_params[index++];
      
      // need to add a new synthetic data point to the observations for the current scenario
      
      unsigned int num_data = this->m_observations.sizeGlobal();
      
      std::vector<double> cat_params;
      switch(model_map[m_cat_model])
        {
          case 0: // constant
            cat_params.push_back(0.08);
            break;
          case 3: // reduced_arrhenius
            cat_params.push_back(0.0285);
            break;
          case 4: // reduced_pwr
            cat_params.push_back(0.218);
            cat_params.push_back(-0.933);
            break;
          default:
            libmesh_error_msg("Unsupported cat model: "+m_cat_model);
            break;
        }
      
      double value = m_grins_eval->calc_absorption_grins(cat_params,m_p,m_Xo,m_Tf);
      
      QUESO::GslVector sample( m_noise.imageSet().vectorSpace().zeroVector() );
      m_noise.realizer().realization(sample);
      double data_point = value + sample[0];
      
      const_cast<V &>(this->m_observations)[num_data-1] = data_point;
    }
  
  private:
    double m_p;
    double m_Xo;
    double m_Tf;
    
    bool m_fix_p;
    bool m_fix_Xo;
    bool m_fix_Tf;

    const std::string & m_cat_model;
    std::shared_ptr<GRINSEvaluator> m_grins_eval;

    const std::vector<std::vector<double>> m_initial_scenarios;
    const QUESO::GaussianVectorRV<V,M> & m_noise;

};

template<typename V = QUESO::GslVector, typename M = QUESO::GslMatrix>
class SIPLikelihood :  public QUESO::GaussianLikelihoodDiagonalCovariance<V,M>
{
  public:

    SIPLikelihood( const QUESO::VectorSet<V,M> & domain,
                    const V & observations,
                    const V & covariance,
                    std::shared_ptr<GRINSEvaluator> & grins_eval,
                    const std::vector<std::vector<double>> initial_scenarios)
    : QUESO::GaussianLikelihoodDiagonalCovariance<V,M>("exp_like_",domain,observations,covariance),
      m_grins_eval(grins_eval),
      m_initial_scenarios(initial_scenarios)
    {}

    virtual void evaluateModel(const V & domainVector, V & modelOutput) const override
    {
      unsigned int num_data = this->m_observations.sizeGlobal();
    
      std::vector<double> cat_params;
      for (unsigned int c=0; c< domainVector.sizeGlobal(); ++c)
        cat_params.push_back(domainVector[c]);
  
      for (unsigned int i = 0; i < num_data; ++i)
      {
        double p  = m_initial_scenarios[i][0];
        double Xo = m_initial_scenarios[i][1];
        double Tf = m_initial_scenarios[i][2];
        
        double ev = m_grins_eval->calc_absorption_surrogate(cat_params,p,Xo,Tf,this->m_env);
        
        modelOutput[i] = ev;
      }

    }
    
    virtual double lnValue( const V & domainVector,
                            const V * domainDirection,
                            V * gradVector,
                            M * hessianMatrix,
                            V * hessianEffect) const override
    {
      return QUESO::GaussianLikelihoodDiagonalCovariance<V,M>::lnValue(domainVector);
    }
  
  private:
    std::shared_ptr<GRINSEvaluator> m_grins_eval;
    const std::vector<std::vector<double>> m_initial_scenarios;


};

template<typename V = QUESO::GslVector, typename M = QUESO::GslMatrix>
class LikelihoodInterface : public QUESO::ExperimentalLikelihoodInterface<V,M>
{
public:

  LikelihoodInterface()
  {
  
  }

  virtual void reinit(std::vector<double> & scenario_params, QUESO::BaseScalarFunction<V,M> & likelihood) override
  {
    dynamic_cast<Likelihood<V,M> &>(likelihood).setScenario(scenario_params);
  }

};


template<typename T>
void get_var_value(const GetPot & input, T & value, std::string input_var, T default_value)
{
  if (input.have_variable(input_var))
    value = input(input_var, default_value);
  else
    libmesh_error_msg("ERROR: Could not find required input parameter: "+input_var);
}

template void get_var_value<std::string>(const GetPot &, std::string &, std::string, std::string);
template void get_var_value<double>(const GetPot &, double &, std::string, double);
template void get_var_value<unsigned int>(const GetPot &, unsigned int &, std::string, unsigned int);


int main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

    std::string dim = std::to_string(2);
    
    std::string metric_string(argv[2]);
    
    std::string queso_input_file;
    if (metric_string == "eig")
      queso_input_file = "./queso_eig.in";
    else
      queso_input_file = "./queso_mh.in";

    QUESO::FullEnvironment env(MPI_COMM_WORLD,queso_input_file, "", NULL);
    
    GetPot input(argv[1]);
    
    std::string cat_model;
    get_var_value<std::string>(input,cat_model,"GridSearchExperimentalDesign/catalycity_model","");

    unsigned int n_sip_params = 0;
    std::vector<double> param_min;
    std::vector<double> param_max;
    std::vector<double> initials;
    std::vector<double> prop_cov;
    
    switch(model_map[cat_model])
      {
        case 0:
          n_sip_params = 1;
          param_min.push_back(0.010);
          param_max.push_back(0.200);
          initials.push_back(0.040);
          prop_cov.push_back(0.02);
          break;
        case 1:
          n_sip_params = 2;
          param_min.push_back(0.01);
          param_min.push_back(-1400.0);
          param_max.push_back(0.10);
          param_max.push_back(-700.0);
          initials.push_back(0.07);
          initials.push_back(-1250.0);
          prop_cov.push_back(0.006);
          prop_cov.push_back(50.0);
          break;
        case 2:
          n_sip_params = 3;
          param_min.push_back(0.01);
          param_min.push_back(700.0);
          param_min.push_back(-1.5);
          param_max.push_back(0.10);
          param_max.push_back(1400.0);
          param_max.push_back(-0.1);
          initials.push_back(0.05);
          initials.push_back(1250.0);
          initials.push_back(-0.75);
          prop_cov.push_back(0.02);
          prop_cov.push_back(50.0);
          prop_cov.push_back(0.10);
          break;
        case 3: // reduced arrhenius
          n_sip_params = 1;
          param_min.push_back(0.01);
          param_max.push_back(0.10);
          initials.push_back(0.05);
          prop_cov.push_back(0.006);
          
          break;
        case 4: // reduced pwr
          n_sip_params = 2;
          param_min.push_back(0.0);
          param_min.push_back(-2.5);
          param_max.push_back(1.0);
          param_max.push_back(-0.1);
          initials.push_back(0.5);
          initials.push_back(-0.75);
          prop_cov.push_back(0.05);
          prop_cov.push_back(0.10);
          break;
        default:
          std::cerr <<"********* Unrecognized catalycity model: " <<cat_model <<std::endl;
          return -1;
      }

//////////////////////////////////////////////
/////////// Surrogate ////////////////////////
//////////////////////////////////////////////

    
    std::string input_filename = "./"+dim+"d/"+cat_model+"/800.in";
    std::string surrogate_filename = "./"+dim+"d/"+cat_model+"/surrogate_data/surr_pTXo.dat";
    std::string surrogate_prefix = "surr_pXoTf";

    std::shared_ptr<QUESO::InterpolationSurrogateIOASCII<QUESO::GslVector,QUESO::GslMatrix>> data_reader(new QUESO::InterpolationSurrogateIOASCII<QUESO::GslVector,QUESO::GslMatrix>());

    data_reader->read(surrogate_filename,env,surrogate_prefix);
    std::shared_ptr<QUESO::LinearLagrangeInterpolationSurrogate<QUESO::GslVector,QUESO::GslMatrix>> surrogate(new QUESO::LinearLagrangeInterpolationSurrogate<QUESO::GslVector,QUESO::GslMatrix>( data_reader->data() ));
    
    std::shared_ptr<GRINSEvaluator> grins_eval(new GRINSEvaluator(input_filename,cat_model,surrogate,data_reader,env.subComm().Comm()));
        
    
//////////////////////////////////////////////
/////////// Noise ////////////////////////////
//////////////////////////////////////////////

    if (env.fullRank() == 0)
      std::cout <<"======================================" <<std::endl
                <<"Setting up the Noise parameters" <<std::endl
                <<"======================================" <<std::endl;
    
    double sigma_noise;
    get_var_value<double>(input,sigma_noise,"GridSearchExperimentalDesign/sigma_noise",0.0);
    
    QUESO::VectorSpace<QUESO::GslVector,QUESO::GslMatrix> noise_space( env, "noise_", 1, NULL );
    
    QUESO::GslVector noise_min( noise_space.zeroVector() );
    noise_min[0] = -1.0e-2;
    
    QUESO::GslVector noise_max( noise_space.zeroVector() );
    noise_max[0] = 1.0e-2;
    
    QUESO::GslVector noise_mean(noise_space.zeroVector());
    noise_mean.cwSet(0.0);
    
    QUESO::GslVector noise_var(noise_space.zeroVector());
    noise_var.cwSet(sigma_noise*sigma_noise);
    
    QUESO::BoxSubset<QUESO::GslVector,QUESO::GslMatrix> noise_domain( "noise_domain_", noise_space, noise_min, noise_max );
    QUESO::GaussianVectorRV<QUESO::GslVector,QUESO::GslMatrix> noise_rv( "noise_rv_", noise_domain, noise_mean, noise_var );
    

//////////////////////////////////////////////
/////////// Data /////////////////////////////
//////////////////////////////////////////////

    if (env.fullRank() == 0)
      std::cout <<"======================================" <<std::endl
                <<"Setting up the Data" <<std::endl
                <<"======================================" <<std::endl;

    unsigned int n_obs;
    get_var_value<unsigned int>(input,n_obs,"GridSearchExperimentalDesign/num_initial_experiments",0);
    
    unsigned int n_data = n_obs + 1;

    QUESO::VectorSpace<QUESO::GslVector,QUESO::GslMatrix> initial_data_space( env, "data_", n_obs, NULL );
    QUESO::VectorSpace<QUESO::GslVector,QUESO::GslMatrix> data_space( env, "data_", n_data, NULL );
    
    QUESO::GslVector initial_variance( initial_data_space.zeroVector() );
    initial_variance.cwSet(sigma_noise*sigma_noise);
    
    QUESO::GslVector variance( data_space.zeroVector() );
    variance.cwSet(sigma_noise*sigma_noise);
    
    std::vector<std::vector<double>> initial_scenarios(n_obs);
    
    QUESO::GslVector sample( noise_space.zeroVector() );
    QUESO::GslVector initial_obs( initial_data_space.zeroVector() );
    QUESO::GslVector obs( data_space.zeroVector() );
    
    std::vector<double> cat_params;
      switch(model_map[cat_model])
        {
          case 0: // constant
            cat_params.push_back(0.08);
            break;
          case 3: // reduced_arrhenius
            cat_params.push_back(0.0285);
            break;
          case 4: // reduced_pwr
            cat_params.push_back(0.218);
            cat_params.push_back(-0.933);
            break;
          default:
            libmesh_error_msg("Unsupported cat model: "+cat_model);
            break;
        }
    
    for (unsigned int s=0; s<n_obs; ++s)
      {
        double p;
        get_var_value<double>(input,p,"GridSearchExperimentalDesign/Experiment"+std::to_string(s)+"/p",0.0);
        
        double Tf;        
        get_var_value<double>(input,Tf,"GridSearchExperimentalDesign/Experiment"+std::to_string(s)+"/Tf",0.0);
        
        double Xo;
        get_var_value<double>(input,Xo,"GridSearchExperimentalDesign/Experiment"+std::to_string(s)+"/Xo",0.0);
        
        initial_scenarios[s].push_back(p);
        initial_scenarios[s].push_back(Xo);
        initial_scenarios[s].push_back(Tf);
        
        double abs = grins_eval->calc_absorption_grins(cat_params,p,Xo,Tf);
        
        noise_rv.realizer().realization(sample);
        obs[s] = abs + sample[0];
        initial_obs[s] = abs + sample[0];
      }



//////////////////////////////////////////////
/////////// SIP Parameters ///////////////////
//////////////////////////////////////////////

    if (env.fullRank() == 0)
      std::cout <<"======================================" <<std::endl
                <<"Setting up the Inverse Problem parameters" <<std::endl
                <<"======================================" <<std::endl;

    QUESO::VectorSpace<QUESO::GslVector,QUESO::GslMatrix> param_space( env, "param_", n_sip_params, NULL );

    QUESO::GslVector min_vals( param_space.zeroVector() );
    for (unsigned int i=0; i<n_sip_params; ++i)
      min_vals[i] = param_min[i];

    QUESO::GslVector max_vals( param_space.zeroVector() );
    for (unsigned int i=0; i<n_sip_params; ++i)
      max_vals[i] = param_max[i];


    QUESO::BoxSubset<QUESO::GslVector,QUESO::GslMatrix> param_domain( "param_domain_", param_space, min_vals, max_vals );


//////////////////////////////////////////////
/////////// Run SIP for Current Belief ///////
//////////////////////////////////////////////

    if (env.fullRank() == 0)
      std::cout <<"======================================" <<std::endl
                <<"Running a SIP to get current param beliefs" <<std::endl
                <<"======================================" <<std::endl;
   
    QUESO::UniformVectorRV<QUESO::GslVector,QUESO::GslMatrix> sip_prior( "sip_prior_", param_domain);
    
    QUESO::GenericVectorRV<QUESO::GslVector,QUESO::GslMatrix> sip_post("sip_post_", param_space);
    
    QUESO::GslVector paramInitials( param_space.zeroVector() );
    for (unsigned int i=0; i<n_sip_params; ++i)
      paramInitials[i] = initials[i];

    QUESO::GslMatrix propCovMatrix(param_space.zeroVector());
    for (unsigned int i=0; i<n_sip_params; ++i)
      propCovMatrix(i,i) = prop_cov[i];

    SIPLikelihood<QUESO::GslVector,QUESO::GslMatrix> sip_likelihood( param_domain, initial_obs, initial_variance, grins_eval, initial_scenarios );

    QUESO::StatisticalInverseProblem<QUESO::GslVector,QUESO::GslMatrix> ip("", NULL,
                                                                          sip_prior,
                                                                          sip_likelihood,
                                                                          sip_post);

    if (metric_string == "eig")
      {
        std::cout <<"\n\n Initial SIP with EIG\n\n";
        ip.solveWithBayesMLSampling();
      }
    else
      {
        std::cout <<"\n\n Initial SIP with MinVar\n\n";
        ip.solveWithBayesMetropolisHastings(NULL, paramInitials, &propCovMatrix);
      }

    QUESO::GslVector post_mean( ip.chain().unifiedMeanPlain() );
    post_mean.mpiBcast(0,env.fullComm());
    
    QUESO::GslVector post_var( ip.chain().unifiedSampleVariancePlain() );
    post_var.mpiBcast(0,env.fullComm());

    env.fullComm().Barrier();

    std::shared_ptr<QUESO::BaseVectorRV<QUESO::GslVector,QUESO::GslMatrix>> prior( new QUESO::GaussianVectorRV<QUESO::GslVector,QUESO::GslMatrix>("exp_prior",param_domain,post_mean,post_var) );

//////////////////////////////////////////////
/////////// Scenario /////////////////////////
//////////////////////////////////////////////

    if (env.fullRank() == 0)
      std::cout <<"======================================" <<std::endl
                <<"Setting up the Scenario parameters" <<std::endl
                <<"======================================" <<std::endl;

    
    unsigned int num_fixed = 0;

    bool fix_p  = false;
    double p_ref = 0.0;
    if (input.have_variable("GridSearchExperimentalDesign/fix_p"))
      {
        fix_p = true;
        num_fixed++;
        get_var_value<double>(input,p_ref,"GridSearchExperimentalDesign/fix_p",-1.0);
        
        if (env.fullRank() == 0)
          std::cout <<"**** Fixing p at " <<p_ref <<std::endl;
      }
    bool fix_Xo = false;
    double Xo_ref = 0.0;
    if (input.have_variable("GridSearchExperimentalDesign/fix_Xo"))
      {
        fix_Xo = true;
        num_fixed++;
        get_var_value<double>(input,Xo_ref,"GridSearchExperimentalDesign/fix_Xo",-1.0);
        
        if (env.fullRank() == 0)
          std::cout <<"**** Fixing Xo at " <<Xo_ref <<std::endl;
      }
    bool fix_Tf = false;
    double Tf_ref = 0.0;
    if (input.have_variable("GridSearchExperimentalDesign/fix_Tf"))
      {
        fix_Tf = true;
        num_fixed++;
        get_var_value<double>(input,Tf_ref,"GridSearchExperimentalDesign/fix_Tf",-1.0);
        
        if (env.fullRank() == 0)
          std::cout <<"**** Fixing Tf at " <<Tf_ref <<std::endl;
      }


    unsigned int n_scenario_params = 3-num_fixed; // p, Xo, Tf

    QUESO::VectorSpace<QUESO::GslVector,QUESO::GslMatrix> scenario_space( env, "scenario_", n_scenario_params, NULL );
    
    unsigned int index = 0;
    QUESO::GslVector scenario_min( scenario_space.zeroVector() );
    QUESO::GslVector scenario_max( scenario_space.zeroVector() );
    
    if (!fix_p) { scenario_min[index] = 200.0; scenario_max[index++] = 350.0; }
    
    if (!fix_Xo) { scenario_min[index] = 0.005; scenario_max[index++] = 0.020; }
    
    if (!fix_Tf) { scenario_min[index] = 600.0; scenario_max[index++] = 1200.0; }
    
    QUESO::BoxSubset<QUESO::GslVector,QUESO::GslMatrix> scenario_domain( "scenario_domain_", scenario_space, scenario_min, scenario_max );
    
    std::vector<unsigned int> n_points;
    if (!fix_p) n_points.push_back(4);
    if (!fix_Xo) n_points.push_back(4);
    if (!fix_Tf) n_points.push_back(4);
    

//////////////////////////////////////////////
/////////// Likelihood ///////////////////////
//////////////////////////////////////////////

    if (env.fullRank() == 0)
      std::cout <<"======================================" <<std::endl
                <<"Setting up the Likelihood" <<std::endl
                <<"======================================" <<std::endl;    
 
    Likelihood<QUESO::GslVector,QUESO::GslMatrix> * like = new Likelihood<QUESO::GslVector,QUESO::GslMatrix>(param_domain, obs, variance, cat_model, grins_eval, initial_scenarios, noise_rv);
    
    if (fix_p)
      like->fix_p(p_ref);
      
    if (fix_Xo)
      like->fix_Xo(Xo_ref);
      
    if (fix_Tf)
      like->fix_Tf(Tf_ref);
    
    std::shared_ptr<QUESO::BaseScalarFunction<QUESO::GslVector,QUESO::GslMatrix>> likelihood(like);
    
    std::shared_ptr<QUESO::ExperimentalLikelihoodInterface<QUESO::GslVector,QUESO::GslMatrix>> interface( new LikelihoodInterface<QUESO::GslVector,QUESO::GslMatrix>() );
    
//////////////////////////////////////////////
/////////// Experimental Design //////////////
//////////////////////////////////////////////

    if (env.fullRank() == 0)
      std::cout <<"======================================" <<std::endl
                <<"Setting up the Experimental Design classes" <<std::endl
                <<"======================================" <<std::endl;  
    
    std::shared_ptr<QUESO::ExperimentMetricBase<QUESO::GslVector,QUESO::GslMatrix>> metric;
    
    if (metric_string == "eig")
      metric.reset( new QUESO::ExperimentMetricEIG<QUESO::GslVector,QUESO::GslMatrix>() );
    else if (metric_string == "min_var")
      metric.reset( new QUESO::ExperimentMetricMinVariance<QUESO::GslVector,QUESO::GslMatrix>(paramInitials,propCovMatrix) );
    else
      libmesh_error_msg("Unknown metric_string: "+metric_string);
    
    if (env.fullRank() == 0)
      std::cout <<"**** Using metric: " <<metric_string <<std::endl;
    
    std::shared_ptr<QUESO::ExperimentalLikelihoodWrapper<QUESO::GslVector,QUESO::GslMatrix>> wrapper( new QUESO::ExperimentalLikelihoodWrapper<QUESO::GslVector,QUESO::GslMatrix>(likelihood,interface) );
    
    std::shared_ptr<QUESO::ScenarioRunner<QUESO::GslVector,QUESO::GslMatrix>> runner( new QUESO::ScenarioRunner<QUESO::GslVector,QUESO::GslMatrix>(prior,wrapper,metric) );
    
    QUESO::GridSearchExperimentalDesign<QUESO::GslVector,QUESO::GslMatrix> exp_design(scenario_domain,n_points,runner);

//////////////////////////////////////////////
/////////// Get Results //////////////////////
//////////////////////////////////////////////

    if (env.fullRank() == 0)
      std::cout <<"======================================" <<std::endl
                <<"Running the Experimental Design" <<std::endl
                <<"======================================" <<std::endl;
    
    QUESO::GslVector experimental_params(param_space.zeroVector());
    
    std::string prefix(argv[3]);
    prefix = prefix+"_"+metric_string;
    
    exp_design.run(experimental_params,prefix);

//////////////////////////////////////////////
/////////// Print Results ////////////////////
//////////////////////////////////////////////

    if (env.fullRank() == 0)
      std::cout <<"======================================" <<std::endl
                <<"Printing the Experimental Design results" <<std::endl
                <<"======================================" <<std::endl;
    
    if (env.fullRank() == 0)
      {
        unsigned int index = 0;
        double p = (fix_p)  ? p_ref  : experimental_params[index++];
        double x = (fix_Xo) ? Xo_ref : experimental_params[index++];
        double t = (fix_Tf) ? Tf_ref : experimental_params[index++];
        
        std::cout <<"\n==================================================" <<std::endl
                  <<"Experimental Design results: " <<std::endl
                  <<"p  = " <<p <<std::endl
                  <<"Xo = " <<x <<std::endl
                  <<"Tf = " <<t <<std::endl
                  <<"==================================================" <<std::endl;
      }
  MPI_Finalize();

  return 0;
}

