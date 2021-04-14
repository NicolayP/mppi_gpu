#ifndef __CONTROLLER_BASE_CLASS_HPP__
#define __CONTROLLER_BASE_CLASS_HPP__

#include <string>
#include "tensorflow/cc/client/client_session.h"

class ControllerBase {
public:
    ControllerBase (const int k,
                    const int tau,
                    const float dt,
                    const int sDim,
                    const int aDim);

    ~ControllerBase ();

    void next(Tensor x);
    bool setActions(vector<Tensor> actions);
    Status logGraph();

private:
    int mK;
    int mTau;
    int mDt;
    int mSDim;
    int mADim;

    Tensor mMu;
    Tensor mSigma;

    Tensor mU;

    Scope mRoot;

    Tensor noise;

    ClientSession mSess;

    void buildGraph();
    void buildCostGraph();
    void buildModelGraph();
}

#endif
