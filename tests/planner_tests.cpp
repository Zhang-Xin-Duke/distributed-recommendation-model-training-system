#include "eps/cost_planner.hpp"
#include <iostream>
#include <stdexcept>
using namespace eps;
void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(){
  try{
    CostBasedPlanner p({1,0.15,2,0,1,0});
    FeatureInput f{1,0,Pooling::sum,{},{0},{}};
    Check(p.SelectRoute(f)==AccessRoute::cpu_pool,"cold route");
    p.Observe({1,0,Pooling::sum,100,30,1,100,20,0});
    Check(p.SelectRoute(f)==AccessRoute::cpu_pool,"insufficient observations");
    p.Observe({1,0,Pooling::sum,100,30,1,100,20,0});
    Check(p.SelectRoute(f)==AccessRoute::transfer_unique,"cost-based route");
    p.Observe({1,0,Pooling::sum,100,30,1,19,20,0});
    Check(p.SelectRoute(f)==AccessRoute::transfer_unique,"hysteresis");
    p.Observe({1,0,Pooling::sum,100,30,1,5,20,0});
    Check(p.SelectRoute(f)==AccessRoute::cpu_pool,"route adaptation");
    f.pooling=Pooling::sequence;Check(p.SelectRoute(f)==AccessRoute::transfer_unique,"sequence compressed as sum");
    p.SetFreeSlots(1,3);p.SetFreeSlots(2,3);
    std::vector<AdmissionCandidate> rows={
      {{1,1},1,1,10,10,64}, // one-use row: negative benefit
      {{1,2},10,10,10,10,64},
      {{1,3},100,10,10,10,64},
      {{1,3},100,10,10,10,64}, // duplicate proposal must not allocate twice
      {{2,4},10,10,10,10,64}};
    auto plans=p.PlanMigration(rows,{128,128,1,1});
    Check(plans.size()==1 && plans[0].rows.size()==2 && plans[0].estimated_bytes==128,"budget/grouping/reserve");
    Check(plans[0].rows[0].row.key==3 && plans[0].rows[1].row.key==2,"reuse benefit ordering");
    Check(p.PlanMigration(rows,{32,32,1,1}).empty(),"byte limit ignored");
    Check(p.PlanMigration(rows,{128,128,0,1}).empty(),"in-flight limit ignored");
    std::cout<<"PASS cost model, hysteresis, reuse admission and migration budgets\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL planner: "<<e.what()<<'\n';return 1;}
}
