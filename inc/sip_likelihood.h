
#include "../catalycity/grins_evaluator.h"

#include <queso/GaussianLikelihoodDiagonalCovariance.h>
#include <queso/GslVector.h>
#include <queso/GslMatrix.h>
#include <queso/BoxSubset.h>


#ifndef SIP_LIKELIHOOD_H
#define SIP_LIKELIHOOD_H

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

#endif

