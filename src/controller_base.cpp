#include <iostream>
#include "tensorflow/core/framework/tensor.h"

ControllerBase::ControllerBase (const int k,
                                const int tau,
                                const float dt,
                                const int s_dim,
                                const int a_dim)
{
    mK = k;
    mTau = tau;

    mSDim = s_dim;
    mADim = a_dim;

    mDt = dt;

    mMu = Tensor(mu, DT_FLOAT);
    mSigma = Tensor(sigma, DT_FLOAT);

    buildModelGraph();
    buildCostGraph();
    buildGraph();
}

ControllerBase::~ControllerBase() {}

bool ControllerBase::setActions(vector<Tensor> actions) {

}

void ControllerBase::next(Tensor x) {
    /*TF_CHECK_OK(sess.Run({{mStateInput, x}, {mActionInput, mU}},
                         {mUpdate},
                         &out_tensor));
    */

}

void ControllerBase::buildModelGraph() {

}

void ControllerBase::buildCostGraph() {

}

ControllerBase::buildGraph() {
    mRoot = Scope::NewRootScope();
    // Input placeholder for the state and the action sequence.
    //auto mStateInput = Placeholder(mRoot.WithOpName("state_input"), DT_FLOAT);
    //auto mActionInput = Placeholder(mRoot.WithOpName("action_sequence"), DT_FLOAT);

    // Generate Random noise for the rollouts
    auto rng = RandomNormal(mRoot.WithOpName("random_number_generation"),
                              Shape({mK, mTau}),
                              DT_FLOAT,
                              RandomNormal::Seed(0));
    noise = AddV2(mRoot.WithOpName("Scale_random"), rng, mMu);

    /*
    // Simulate the model for all the samples and compute the cost of each model
    auto sim = Simulate(mRoot.WithOpName("simulation"), s_input, a_input);

    /* Compute the min of the cost for numerical stability. (I.e at least one
       sample with a none 0 weight. * /
    auto beta = Min(mRoot.WithOpName("Min"), sim);

    // Exponential of the path cost.
    auto exp = Power(mRoot.WithOpName("Exponential"), sim, beta);

    // Normalisation term.
    auto nabla = Sum(mRoot.WithOpName("normalisation_factor"), exp);

    // Compute the path weights.
    auto weights = Div(mRoot.WithOpName("weights"), exp, nabla);

    // Update the action sequence with the weighted average.
    auto mUpdate = Mean(mRoot.WithOpName("Update_actions"), weights, noise, axis=0);
    */
}

void ControllerBase::run() {
    mSess(mRoot);
    TF_CHECK_OK(mSess.Run({{}},
                          {noise}, &outTensor));
}

Status ControllerBase::logGraph() {
    GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
    SummaryWriterInterface* w;
    TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "graphs",
                                        ".img-graph", Env::Default(), &w));
    TF_CHECK_OK(w->WriteGraph(0, make_unique<GraphDef>(graph)));

    return Status::OK();
}
