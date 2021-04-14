## Tensorflow cpp installation

1. Installed golang. Just download binary and set the different env variables
    '''
    export GOROOT=/path/to/install
    export GOPATH=$HOME/path/to/workspace
    export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
    export PATH=$PATH:$(go env GOPATH)/bin
    '''
2. Install bazelisk, a wrapper around bazel that will find the correct bazel version for your Tensorflow requirements.

3. Clone Tensorflow repo.
4. Install Cuda, cuda toolkit and cudNN for GPU support. Here we want tensorflow 2.1 so we need cuda 10.1, cudNN 7.6.
 
