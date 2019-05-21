#include <map>

#ifndef MODEL_MAP
#define MODEL_MAP

static std::map<std::string,unsigned int> model_map = {{"constant",0}, {"arrhenius",1}, {"pwr",2}, {"reduced_arrhenius",3}, {"reduced_pwr",4}};

#endif
