#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_IN 2
#define NUM_HIDD 2
#define NUM_OUT 1
#define NUM_TRAIN_SETS 4


double sigmoid(double x) {


    return 1.0 / (1.0 + exp(-x));

}

double d_sigmoid(double x) {


    return x * (1.0 - x); 


}

//random weight 
double init_weight() {


    return ((double)rand() / ((double)RAND_MAX)) * 2.0 - 1.0;


}

int main()
 {

    srand((unsigned int)time(NULL));

    
    const double inputs[NUM_TRAIN_SETS][NUM_IN] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    const double targets[NUM_TRAIN_SETS][NUM_OUT] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    
    double w_hidd[NUM_IN][NUM_HIDD];
    double b_hidd[NUM_HIDD];
    double w_out[NUM_HIDD][NUM_OUT];
    double b_out[NUM_OUT];

    // Initialize weights and biases randomly
    for (int i = 0; i < NUM_IN; i++) {
        for (int j = 0; j < NUM_HIDD; j++) {
            w_hidd[i][j] = init_weight();
        }
    }
    for (int j = 0; j < NUM_HIDD; j++) {
        b_hidd[j] = init_weight();
        for (int k = 0; k < NUM_OUT; k++) {
            w_out[j][k] = init_weight();
        }
    }
    for (int k = 0; k < NUM_OUT; k++) {
        b_out[k] = init_weight();
    }

    
    const double lr = 0.5;          // Llern rate
    const int epochs = 25000;       // epoch count

    // Training Loop
    for (int epoch = 0; epoch <= epochs; epoch++) {
        double total_loss = 0.0;

        for (int p = 0; p < NUM_TRAIN_SETS; p++)
         {


            // Forward pass von hidd
            double hidden_layer[NUM_HIDD];
            for (int j = 0; j < NUM_HIDD; j++) {
                double activation = b_hidd[j];
                for (int i = 0; i < NUM_IN; i++) {
                    activation += inputs[p][i] * w_hidd[i][j];
                }
                hidden_layer[j] = sigmoid(activation);
            }

            // Forward pass von out
            double output_layer[NUM_OUT];
            for (int k = 0; k < NUM_OUT; k++) {
                double activation = b_out[k];
                for (int j = 0; j < NUM_HIDD; j++) {
                    activation += hidden_layer[j] * w_out[j][k];
                }
                output_layer[k] = sigmoid(activation);
            }

            // Compute MSE Loss
            double error = targets[p][0] - output_layer[0];
            total_loss += 0.5 * error * error;

            // Backpropagation von dout
            double delta_output[NUM_OUT];

            for (int k = 0; k < NUM_OUT; k++) 
            {

                delta_output[k] = 
                (output_layer[k] - targets[p][k]) *
                d_sigmoid(output_layer[k]);
            }

        // Backpropagation von dhidd 

            double delta_hidden[NUM_HIDD];
            for (int j = 0; j < NUM_HIDD; j++) 
            {
                double error_sum = 0.0;
                for (int k = 0; k < NUM_OUT; k++) 
                {

                    error_sum += delta_output[k] * w_out[j][k];

                }

                delta_hidden[j] = error_sum * d_sigmoid(hidden_layer[j]);
            }

            //Update Output Layer
            for (int k = 0; k < NUM_OUT; k++) 
            {
                b_out[k] -= lr * delta_output[k];
                for (int j = 0; j < NUM_HIDD; j++) 
                {
                    w_out[j][k] -= lr * delta_output[k] * hidden_layer[j];
                }
                
            }

            //Update Hidden Layer
            for (int j = 0; j < NUM_HIDD; j++) 
            {
                b_hidd[j] -= lr * delta_hidden[j];

                for (int i = 0; i < NUM_IN; i++) 
                {
                    w_hidd[i][j] -= lr * delta_hidden[j] * inputs[p][i];
                }
            }
        }

        if (epoch % 5000 == 0) 
        {
            printf("Epoch %5d | MSE Loss: %.6f\n", epoch, total_loss / NUM_TRAIN_SETS);
        }
    }

    // Testing Final Predictions
    printf("\n---final guess!---\n");
    for (int p = 0; p < NUM_TRAIN_SETS; p++) 
    {

        double hidden_layer[NUM_HIDD];
        for (int j = 0; j < NUM_HIDD; j++) 
        {
            double act = b_hidd[j];

            for (int i = 0; i < NUM_IN; i++) 
            {
                act += inputs[p][i] * w_hidd[i][j];
            }

            hidden_layer[j] = sigmoid(act);
        }

        double final_out = b_out[0];
        for (int j = 0; j < NUM_HIDD; j++) 
        {
            final_out += hidden_layer[j] * w_out[j][0];
        }

        final_out = sigmoid(final_out);

        printf("In: [%.0f, %.0f] -> Target: %.0f | Predicted: %.4f (round: %d)\n",
               inputs[p][0], inputs[p][1], targets[p][0], final_out, final_out >= 0.5 ? 1 : 0);
    }

    return 0;
}