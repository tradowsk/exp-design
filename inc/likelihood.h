
#include "../catalycity/grins_evaluator.h"

#include <queso/GaussianLikelihoodDiagonalCovariance.h>
#include <queso/GslVector.h>
#include <queso/GslMatrix.h>
#include <queso/BoxSubset.h>
#include <queso/GaussianVectorRV.h>
#include <queso/ExperimentalLikelihoodInterface.h>
#include <queso/ExperimentalLikelihoodWrapper.h>


#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H

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

#endif

