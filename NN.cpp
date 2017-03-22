#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

using namespace std;

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
            calcOutputGradients();//calculate output error
            calcHiddenGradients();//calculate hidden layer error
            updateInputWeights();//
    private:
            double transferFunction(double x) {return tanh(x);}//apply the tanh to the output
            double transferFunctionDerivative(double x){return 1.0-x*x;}//tanh derivative aprox 1-x*x (original = 1- (tanh(x)^2)x
            double randomWeight(){ return rand() / double(RAND_MAX); }//return a random weight
            sumDOW();//used to calculate hidden layer error

            unsigned index;
            vector <Connection>outputWeights;
            double outputVal;
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
    private:
        vector <Layer> netLayer;
};
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
        }
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
    for(int i = 0; i< netLayer.back().size()-1;i++)
    {
        outputValue.push_back(netLayer.back()[i].getOutputVal());
    }
}

void Network::backProp(const vector<double> &targetVals)//calculate network error using rms
{
    Layer &outputLayer = netLayer.back();//a reference to the output neurons
    double error = 0;
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
//****************NEURON****************************************//

int main()

{
    vector < int > s;s.push_back(2);s.push_back(4);s.push_back(4);s.push_back(1);//structure of the neural network
    Network network(s);


    vector  < double > data;data.push_back(1.0);data.push_back(0.0);
    vector < double > results;


    network.feedForward(data);
    network.getResults(results);


    return  0;
}


