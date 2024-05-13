#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Define the parameters
const int inputSize = 8;
const int hiddenSize = 3;

// Define the autoencoder class
class Autoencoder 
{
private:
    vector<vector<double>> weightsInputHidden;
    vector<vector<double>> weightsHiddenOutput;
    vector<double> hiddenLayer;
    vector<double> outputLayer;
    vector<double> reconstructedInput;

public:
    Autoencoder() 
    {
        // Initialize weights randomly
        weightsInputHidden.resize(inputSize, vector<double>(hiddenSize));
        weightsHiddenOutput.resize(hiddenSize, vector<double>(inputSize));
        for (int i = 0; i < inputSize; ++i) 
        {
            for (int j = 0; j < hiddenSize; ++j) 
            {
                weightsInputHidden[i][j] = ((double)rand() / RAND_MAX) - 0.5;
                weightsHiddenOutput[j][i] = ((double)rand() / RAND_MAX) - 0.5;
            }
        }
        hiddenLayer.resize(hiddenSize);
        reconstructedInput.resize(inputSize);
    }

    // Sigmoid activation function
    double sigmoid(double x) 
    {
        return 1.0 / (1.0 + exp(-x));
    }

    // Forward pass through the autoencoder
    void forward(const vector<double>& input) 
    {
        // Compute hidden layer activations
        for (int j = 0; j < hiddenSize; ++j) 
        {
            double activation = 0.0;
            for (int i = 0; i < inputSize; ++i)
                activation += input[i] * weightsInputHidden[i][j];
            hiddenLayer[j] = sigmoid(activation);
        }

        // Compute output layer activations (reconstructed input)
        for (int i = 0; i < inputSize; ++i) 
        {
            double activation = 0.0;
            for (int j = 0; j < hiddenSize; ++j) 
                activation += hiddenLayer[j] * weightsHiddenOutput[j][i];
            reconstructedInput[i] = sigmoid(activation);
        }
    }

    // Print the reconstructed input
    void printReconstructedInput() 
    {
        for (int i = 0; i < inputSize; ++i)
            cout << reconstructedInput[i] << " ";
        cout << endl;
    }
};

int main() 
{
    vector<vector<double>> input;
    input = 
    {
        {0,0,0,0,0,0,0,1},
        {0,0,0,0,0,0,1,0},
        {0,0,0,0,0,1,0,0},
        {0,0,0,0,1,0,0,0},
        {0,0,0,1,0,0,0,0},
        {0,0,1,0,0,0,0,0},
        {0,1,0,0,0,0,0,0},
        {1,0,0,0,0,0,0,0}
    };

    for (int i = 0; i < input.size(); ++i) 
    {
        cout << "For input: ";
        for (int j = 0; j < input[i].size(); ++j)
            cout << input[i][j]; 
        cout << endl;

        Autoencoder autoencoder;
        autoencoder.forward(input[i]);
        autoencoder.printReconstructedInput();

        cout << endl;
    }

    return 0;
}