# include "NN.h"
# include <vector>
#include <iostream>


using namespace std;

int main()
{


    vector < int > s;s.push_back(2);s.push_back(4);s.push_back(4);s.push_back(1);//structure of the neural network
    Network network(s);


    vector  < double > data;data.push_back(1.0);data.push_back(0.0);
    vector < double > results;
    vector < double > target; target.push_back(1);


    for(int a=0;a<10000;a++)
        {
    network.feedForward(data);
    network.getResults(results);
    network.backProp(target);
    network.showNetworkData(results,target);

    cout << "Net recent average error: "
        << network.getRecentAverageError() << endl;
        }



return 0;
}
