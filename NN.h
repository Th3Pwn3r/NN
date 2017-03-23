#ifndef NN.h

#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

using namespace std;

double eta = 0.15;    // overall net learning rate, [0.0..1.0]
double alpha = 0.5;

/********************Layers of neurons used by Network class*******************************/
class Neuron;
typedef vector<Neuron> Layer;
/********************Layers of neurons used by Network class*******************************/

/********************Hold the weights from and for each neuron in the net******************/
struct Connection
{
  double weight;//a weight to a specific neuron
  double deltaWeight;//it's used to apply momentum to the weight change
};
/********************Hold the weights from and for each neuron in the net******************/

/********************Each of the individual neuron plus some computation*****************/
class Neuron
{
    public:

            Neuron(int index, int connections);//default constructor
            void setOutputVal(double val) {outputVal = val;}// Manage output values
            double getOutputVal()const {return outputVal;}
            void feedForward(const Layer &prevLayer);// Add his weighted value to the next layer
            void calcOutputGradients(double targetVal);//calculate output error
            void calcHiddenGradients(const Layer &nextLayer);//calculate hidden layer error
            void updateInputWeights(Layer &prevLayer);//
    private:
            double transferFunction(double x) {return tanh(x);}//apply the tanh to the output and apply the transfer function
            double transferFunctionDerivative(double x){return 1.0-x*x;}//tanh derivative aprox 1-x*x (original = 1- (tanh(x)^2)x
            double randomWeight(){ return rand() / double(RAND_MAX); }//return a random weight
            double sumDOW(const Layer &nextLayer)const;//used to calculate hidden layer error

            unsigned index;
            vector <Connection>outputWeights;
            double outputVal;
            double gradient;

};
/********************Each of the individual neuron plus some computation******************/

/********************The neuron holder****************************************************/
class Network
{
    public:
        Network();//default constructor that search a file
        Network(const vector <int> &structure);//this constructor get the structure
        void feedForward(const vector<double> &inputData);
        void backProp(const vector<double> &targetVals);
        void getResults(vector < double > &outputValue)const;
        double getRecentAverageError()const{return recentAverageError;};

        void showNetworkData(const vector <double> &results,const vector <double> &target);

    private:
        vector <Layer> netLayer;
        double error;
        double recentAverageError;
        static double recentAverageSmoothingFactor;
};
double Network::recentAverageSmoothingFactor = 0.1; // Number of training samples to average over
/********************The neuron holder****************************************************/

//*****************************************************************IMPLEMENTATION************************************************************************//

//****************NETWORK****************************************//
Network::Network(const vector <int> &structure)
{
    for(int i=0;i<structure.size();i++)
    {
        int nextLayerSize =  i == structure.size()-1 ? 0 : structure[i+1];

        netLayer.push_back(Layer());
        for(int n=0;n<=structure[i];n++)
        {
        netLayer[i].push_back(Neuron(n,nextLayerSize));

        #ifdef DEBUG
        cout<<"Neuron :"<<n<<" @ layer : "<<i<<endl;
        #endif // DEBUG
        }netLayer.back().back().setOutputVal(1.0);
    }



}

void Network::feedForward(const vector<double> &inputData)
{
    //check the lenght that must be equal
    assert(inputData.size() == netLayer[0].size() - 1);

    //assign the data to the input of the network
    for (int n=0; n<inputData.size();n++)
    {
        netLayer[0][n].setOutputVal(inputData[n]);
    }

    //propagate
    for(int i=1;i<netLayer.size();i++)
    {
        //Prev layer ref
        Layer &prev = netLayer[i-1];

        //for each neuron of current layer, feed forward
        for(int n = 0; n<netLayer[i].size()-1;n++)
        {
            netLayer[i][n].feedForward(prev);
        }
    }
}

void Network::getResults(vector < double > &outputValue) const
{
    outputValue.clear();
    for(int i = 0; i< netLayer.back().size()-1;i++)
    {
        outputValue.push_back(netLayer.back()[i].getOutputVal());
    }
}

void Network::backProp(const vector<double> &targetVals)//calculate network error using rms
{
    Layer &outputLayer = netLayer.back();//a reference to the output neurons
    error = 0;

    for (int n = 0; n < outputLayer.size() - 1; n++) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();//get error for each output neuron
        error += delta * delta;//sum the square of the error to the error pool
    }

    error /= outputLayer.size() - 1; // get average error squared
    error = sqrt(error); // RMS

    // Implement a recent average measurement
    recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for (int n = 0; n < outputLayer.size() - 1; n++) {
        outputLayer[n].calcOutputGradients(targetVals[n]);//calculate the output error scaled by the transfer function
    }

    // Calculate hidden layer gradients
    for (int l = netLayer.size() - 2; l > 0; l--) {
        Layer &hiddenLayer = netLayer[l];
        Layer &nextLayer = netLayer[l + 1];

        for (int n = 0; n < hiddenLayer.size(); n++) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (int layerNum = netLayer.size() - 1; layerNum > 0; layerNum--) {
        Layer &layer = netLayer[layerNum];
        Layer &prevLayer = netLayer[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; n++) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Network::showNetworkData(const vector <double> &results,const vector <double> &target)
    {
        cout<<endl<<"Network info :"<<endl;
        cout<<"ETA : "<<eta<<endl;
        cout<<"Alpha : "<<alpha<<endl;
        cout<<"Input data : ";


        for(int i = 0; i< netLayer[0].size()-1;i++)
        {
        cout<<netLayer[0][i].getOutputVal()<<" ";
        }

        cout<<endl<<"Output  : ";
        for(int i = 0; i< netLayer.back().size()-1;i++)
        {
        cout<<results[i]<<" ";
        }

        cout<<endl<<"Expected value : ";

       for(int i = 0; i<target.size();i++)
       {
        cout<<target[i]<<" ";
       }cout<<endl;

    }
//****************NETWORK****************************************//

//****************NEURON****************************************//
Neuron::Neuron(int indexx, int connections)
{
#ifdef DEBUG
cout<<"Next neuron will connect to : "<<connections<<" neurons."<<endl;
#endif // DEBUG


 for (unsigned c = 0; c < connections; c++) {
        outputWeights.push_back(Connection());
        outputWeights.back().weight = randomWeight();
    }

    index = indexx;
}

void Neuron::feedForward(const Layer &prevLayer)//const??
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.j

    for (int n = 0; n < prevLayer.size(); n++) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].outputWeights[index].weight;
    }
    //this uses the tanh
    outputVal = Neuron::transferFunction(sum);

    #ifdef DEBUG
    cout<<"Output of neuron : "<<index<<" - "<<outputVal<<endl;
    #endif // DEBUG
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - outputVal;
    gradient = delta * Neuron::transferFunctionDerivative(outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    gradient = dow * Neuron::transferFunctionDerivative(outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer)const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += outputWeights[n].weight * nextLayer[n].gradient;
    }

    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); n++) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.outputWeights[index].deltaWeight;

// Individual input, magnified by the gradient and train rate:
// Also add momentum = a fraction of the previous delta weight;
        double newDeltaWeight = eta * neuron.getOutputVal() * gradient+ alpha * oldDeltaWeight;

        neuron.outputWeights[index].deltaWeight = newDeltaWeight;
        neuron.outputWeights[index].weight += newDeltaWeight;
    }
}
//****************NEURON****************************************//


#endif // NN

