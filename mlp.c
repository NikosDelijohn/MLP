/* 
 +---------------------------------------------+
 * MYE035@CSE.UOI - Computational Intelligence *
 * ------------------------------------------- *
 *  prof: Aristidis Lykas                      *
 * ------------------------------------------- *
 *  stud#1: Deligiannis Nikos     - 2681       *
 *  stud#2: Homondozlis Paschalis - 2858       *
 * ------------------------------------------- *
 *  Multilayer Perceptron Implementation       *
 +---------------------------------------------+
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <signal.h>

// Macros
#define CIRCLE_A(x,y) pow(x-1,2) + pow(y-1,2)
#define CIRCLE_B(x,y) pow(x+1,2) + pow(y+1,2)
#define RANDOM_R(A,B) rand()/(double)RAND_MAX * (B-A) + A

// Colors
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KCYN  "\x1B[36m"
#define KYEL  "\x1B[33m"

// MLP Variables
#define d 	2  	    /* Number of inputs  */
#define p   3  	    /* Number of outputs */
#define n   0.005  /* Learning rate     */
#define H   3       /* MLP lvls + output */ 
#define HH  2       /* MLP hidden lvls 	 */
#define H1	7  	    /* First Hidden Level neurons  */
#define H2 	5  	    /* Second Hidden Level neurons */
#define f   0    	/* Activaction Function: Logistic (f=0), Tanh (f=1) */
#define L   300  	/* Number of batches to be updated while learning.  */
#define END -1  	/* Used to determine where the training will stop.  */

#define TOTAL_WEIGHTS  H1*(d+1) + H2*(H1+1) + p*(H2+1)
#define TOTAL_EPOCHS 5000

typedef struct Points
{
	double x_1; 
	double x_2;
	int encoding[3]; /* One-Hot encoding | C1 = [1 0 0] , C2 = [0 1 0] , C3 = [0 0 1] */

}point_t;

typedef struct Neuron
{	

	double *weights;
	double *thetas; 
	double delta;
	double output;
	
}neuron_t; 

typedef struct Network
{

	neuron_t *layer[H];

}net_t;

// Globals
point_t training_set[3000]; /* counts as input layer */
point_t testing_set [3000];
net_t network;

int class_1 = 0;
int class_2 = 0;
int class_3 = 0;

int class_1_t = 0;
int class_2_t = 0;
int class_3_t = 0;

int levels[3] = {H1, H2, p};  

double total_network_thetas[TOTAL_WEIGHTS]; 

// Function Declaration
void print_network();
void dataset_encoding();
void initialize_network();
void stop_training(int signal_number);
void forward_pass(point_t x);
void back_propagation(point_t x);
void training_via_gradient_descent();
void testing();

double activation();
double delta_calculation();


/* Used for debugging purposes. This function
 * prints the whole network stats. For every
 * Layer, its neurons and for every neuron its
 * values (weights,thetas,delta,output)
 */
void print_network()
{   
    int i,j,k;
    int cond;

    printf(" = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = \n");

    for (i = 0; i < H; i++)
    {   
        printf("[LAYER %d]\n",i);
        for(j = 0; j < levels[i]; j++)
        {
            printf("\t[NEURON %d]\n",j);
            cond  = ( i == 0 ) ? 2 : levels[i-1];

            for(k = 0; k <= cond; k++)
                printf("\t\t Weight[%d] = %lf\n",k,network.layer[i][j].weights[k]);

            printf("\t\t -- \n");

            for(k = 0; k <= cond; k++)
                printf("\t\t Theta[%d] = %lf\n",k,network.layer[i][j].thetas[k]);

            printf("\t\t -- \n");

            printf("\t\t Output = %lf\n", network.layer[i][j].output);

            printf("\t\t -- \n");

            printf("\t\t Delta = %lf\n", network.layer[i][j].delta);
        }
    }

     printf(" = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = \n");
}

/* 
 * This function is used to manipulate each point (x1,x2) which is contained 
 * at the "datset.dat" file. It categorizes and encodes the points according
 * to the areas that they belong. '100' stands for C1 points, '010' stands 
 * for C2 points and '001' stands for C3 points.
 */
void dataset_encoding()
{

	int i;
	double x_1,x_2,tmp;

    FILE *fp = fopen("dataset.txt","r");

	for (i = 0; i < 1500; i++)
    {	
    	fscanf(fp,"%lf",&x_1);
    	fscanf(fp,"%lf",&x_2);

    	tmp = CIRCLE_A(x_1,x_2);

    	training_set[i].x_1 = x_1;
    	training_set[i].x_2 = x_2;

    	if (tmp <= 0.16)
    	{

    		training_set[i].encoding[0] = 1;
    		training_set[i].encoding[1] = 0;
    		training_set[i].encoding[2] = 0;
            class_1 ++;

    	}
    	else if (tmp > 0.16 && tmp < 0.64)
    	{

    		training_set[i].encoding[0] = 0;
    		training_set[i].encoding[1] = 1;
    		training_set[i].encoding[2] = 0;
            class_2 ++;
    	}
    	else 
    	{

    		training_set[i].encoding[0] = 0;
    		training_set[i].encoding[1] = 0;
    		training_set[i].encoding[2] = 1;
            class_3 ++;
    	}
    }

    for (i = 0; i < 1500; i++)
    {

    	fscanf(fp,"%lf",&x_1);
    	fscanf(fp,"%lf",&x_2);
    	tmp = CIRCLE_A(x_1,x_2);

    	testing_set[i].x_1 = x_1;
    	testing_set[i].x_2 = x_2;

    	if (tmp <= 0.16)
    	{

    		testing_set[i].encoding[0] = 1;
    		testing_set[i].encoding[1] = 0;
    		testing_set[i].encoding[2] = 0;
            class_1_t ++;

    	}
    	else if (tmp > 0.16 && tmp < 0.64)
    	{

    		testing_set[i].encoding[0] = 0;
    		testing_set[i].encoding[1] = 1;
    		testing_set[i].encoding[2] = 0;
            class_2_t ++;
    	}
    	else 
    	{

    		testing_set[i].encoding[0] = 0;
    		testing_set[i].encoding[1] = 0;
    		testing_set[i].encoding[2] = 1;
            class_3_t ++;

    	}

    }

    for (i = 1500; i < 3000; i++)
    {	
    	fscanf(fp,"%lf",&x_1);
    	fscanf(fp,"%lf",&x_2);
    	tmp = CIRCLE_B(x_1,x_2);

    	training_set[i].x_1 = x_1;
    	training_set[i].x_2 = x_2;

    	if (tmp <= 0.16)
    	{

    		training_set[i].encoding[0] = 1;
    		training_set[i].encoding[1] = 0;
    		training_set[i].encoding[2] = 0;
            class_1 ++;

    	}
    	else if (tmp > 0.16 && tmp < 0.64)
    	{

    		training_set[i].encoding[0] = 0;
    		training_set[i].encoding[1] = 1;
    		training_set[i].encoding[2] = 0;
            class_2 ++;

    	}
    	else 
    	{

    		training_set[i].encoding[0] = 0;
    		training_set[i].encoding[1] = 0;
    		training_set[i].encoding[2] = 1;
            class_3 ++;

    	}
    }

    for (i = 1500; i < 3000; i++)
    {

    	fscanf(fp,"%lf",&x_1);
    	fscanf(fp,"%lf",&x_2);
    	tmp = CIRCLE_B(x_1,x_2);

    	testing_set[i].x_1 = x_1;
    	testing_set[i].x_2 = x_2;

    	if (tmp <= 0.16)
    	{

    		testing_set[i].encoding[0] = 1;
    		testing_set[i].encoding[1] = 0;
    		testing_set[i].encoding[2] = 0;
            class_1_t ++;

    	}
    	else if (tmp > 0.16 && tmp < 0.64)
    	{

    		testing_set[i].encoding[0] = 0;
    		testing_set[i].encoding[1] = 1;
    		testing_set[i].encoding[2] = 0;
            class_2_t++;

    	}
    	else 
    	{

    		testing_set[i].encoding[0] = 0;
    		testing_set[i].encoding[1] = 0;
    		testing_set[i].encoding[2] = 1;
            class_3_t ++;

    	}

    }

    fclose(fp);
}

/*
 * In case someone wants to terminate early the learning 
 * process, the handler will redirect the control and 
 * initiate the testing() process.
 */
void stop_training(int signal_number)
{

    if(signal_number == SIGINT)
    {
        puts("");
        puts("Training stopped by user. Initiating network Testing!");
        testing();
        exit(1);
    }
}

/* 
 * This function initializes the whole MLP network by first allocating all
 * the necessary memory which is needed for our architecture and then by 
 * allocating all the memory needed for every neuron in it. Input layer
 * is not included because its trivial.
 */
void initialize_network()
{	
	int i,j,v,special;
	int num_of_w_i;

	for (i = 0, j = 0; i < H; i++ , j++ )
	{

		network.layer[i] = (neuron_t *) malloc( levels[j] * sizeof(neuron_t) );
		
        if (network.layer[i] == NULL)
			exit(1);

	}

	for (i = 0; i < H; i++) 
	{

		for (j = 0; j < levels[i]; j++ ) 
		{

            special = ( i == 0 ) ? 2 : levels[i-1];

			num_of_w_i = (special + 1 /* for BIAS */ ) * sizeof(double);

            // Allocating thetas.
			network.layer[i][j].thetas = (double *) malloc(num_of_w_i);
			if (network.layer[i][j].thetas == NULL)
				exit(1);

            // Allocating & Initializing weights.
			network.layer[i][j].weights = (double *) malloc(num_of_w_i);
			if (network.layer[i][j].weights == NULL)
				exit(1);

			for (v = 0; v <= special; v++)
				network.layer[i][j].weights[v] = RANDOM_R(-1,1);

		}
	}
}

/*
 * According to the f (which is defined at line 32) 
 * the proper function is used by EVERY neuron in the
 * MLP network.
 */
double activation (double x)
{
    if (f == 0) return (1 / (double) (1 + exp(-x)));
    else        return ((exp(x) - exp(-x)) / (double) (exp(x) + exp(-x))); 
}

/* 
 * This function is used to calculate the delta value
 * of each neuron. Its returning the derivatives of the
 * activation function used in every case.
 */
double delta_calculation ( double x )
{   
    if(f == 0) return (x * (1-x));
    else       return (1 - pow(x, 2));
}


/*
 * Given an input point (x1,x2) it calculates the output vector. 
 */
void forward_pass(point_t x)
{   

    int h,i,j;

    double sum  = 0.0;

    for(h = 0; h < H; h++) 

        for ( i = 0; i < levels[h]; i++)
        {

            switch(h)
            {
                case 0: 

                    sum += network.layer[h][i].weights[0];
                    sum += network.layer[h][i].weights[1] * x.x_1;
                    sum += network.layer[h][i].weights[2] * x.x_2;

                    network.layer[h][i].output = activation(sum);

                    sum = 0.0; // reset

                    break;

                default:

                    sum += network.layer[h][i].weights[0];

                    for(j = 0; j < levels[h-1]; j++)

                        sum += network.layer[h][i].weights[j+1] * \
                               network.layer[h-1][j].output;

                    network.layer[h][i].output = activation(sum);

                    sum = 0.0; // reset

            }

        }       
}

/*
 * Calculation of error (delta) and partial derivative 
 * per weight (thetas) for every neuron in the network.
 */
void back_propagation(point_t x)
{
    int h,i,j;
    int cond;

    double sum = 0.0;
    double tmp;

    //First use forward_pass to find every neuron's output 
    forward_pass(x);

    //Delta & Theta calculation for output layer.
    for(i = 0; i < p; i++) 
    {   
        tmp = delta_calculation(network.layer[2][i].output);
        network.layer[2][i].delta     = tmp * ( network.layer[2][i].output - x.encoding[i] );  
        network.layer[2][i].thetas[0] = network.layer[2][i].delta; // Theta of BIAS.

        // Thetas - {BIAS}
        for(j = 0; j < levels[1]; j++)

            network.layer[2][i].thetas[j+1] = network.layer[2][i].delta * \
                                              network.layer[1][j].output;
    }

    // Delta & Theta calculation for H2 and H1.
    for(h = 1; h >= 0; h--)  

        for(i = 0; i < levels[h]; i++)
        {

            // Sum of weights*deltas
            for(j = 0; j < levels[h+1]; j++)

                sum += network.layer[h+1][j].weights[i+1] * \
                       network.layer[h+1][j].delta;

            tmp = delta_calculation(network.layer[h][i].output);
            network.layer[h][i].delta     = tmp * sum;
            network.layer[h][i].thetas[0] = network.layer[h][i].delta;

            sum = 0.0; // reset

            // Thetas - {BIAS}
            switch(h)
            {
                case 0:

                    network.layer[h][i].thetas[1] = network.layer[h][i].delta * x.x_1;
                    network.layer[h][i].thetas[2] = network.layer[h][i].delta * x.x_2;
                    break;

                default:

                    for(j = 0; j < levels[h-1]; j++)
                        network.layer[h][i].thetas[j+1] = network.layer[h][i].delta * \
                                                          network.layer[h-1][j].output;
            }

        }
}    

/* 
 * The mother of all learning processes
 */
void training_via_gradient_descent()
{
    int epoch,h,i,j;
    int batch,reader;
    int thetas_cc;
    int cond;

    thetas_cc = 0;

    double sum      = 0.0;
    double error    = 0.0;
    double prev_er  = 0.0;

    FILE *fp;
    fp = fopen("error.dat","w+");

    epoch  = 0;
    reader = 0;

    printf("%sTraining: [",KYEL);

    while(epoch != END)
    {   
        //Initializing partial weights.
        for(i = 0; i < TOTAL_WEIGHTS; i++)
        
            total_network_thetas[i] = 0.0;
        
        //Batch routine.
        for(batch = 0; batch < L; batch++)
        {

            back_propagation(training_set[reader]);
            reader++;

            //Getting all thetas.
            for(h = 0; h < H; h++)
            {
                for(i = 0; i < levels[h]; i++)
                {
                    cond = (h == 0) ? 2 : levels[h-1];

                    for(j = 0; j < cond + 1; j++)
                    {
                        total_network_thetas[thetas_cc] += network.layer[h][i].thetas[j];
                        thetas_cc ++;
                    }

                }
            }

            thetas_cc = 0; // reset
        }
  
        thetas_cc = 0;
        //Batch Done.Update the weights.
        for(h = 0; h < H; h++)
        {
            for(i = 0; i < levels[h]; i++)
            {

                cond = (h == 0) ? 2 : levels[h-1];

                for(j = 0; j < cond + 1; j++)
                {
                    network.layer[h][i].weights[j] -= n * total_network_thetas[thetas_cc];
                    thetas_cc++;
                }
            }
        }

        thetas_cc = 0;

        //Square Error Calculation.

        if(reader == 3000)
        {   

        	for (j = 0; j < 3000; j++)
        	{
        		forward_pass(training_set[j]);

        		sum = 0.0;

	            for(i = 0; i < p; i++)
	            
	                sum += pow(training_set[j].encoding[i] - network.layer[2][i].output,2);

	            sum = sum / (double) 2.0;

	            error += sum;
	        }    

            fprintf(fp,"%d %lf\n",epoch,error);
            //printf("Epoch[#%d]: Error %lf\n",epoch,error);

            prev_er = error;            
            reader = 0;  // reset
            epoch ++;    // reset
            error = 0.0; // reset
            sum   = 0.0; // reset 

            // Showcasing Progress
            
            if(epoch % 100 == 0)
            {
                printf("%s#=", KYEL);
                printf("\b");
                fflush(stdout);
            }
            
        }

        // Terminating condition #1.
        epoch = (epoch == TOTAL_EPOCHS) ? epoch = END : epoch; 
        // error difference can be implemented if you want.
        
    }

    printf("]\n");
    fclose(fp);
}

/* 
 * We use the testing_set to measure the generalizing ability
 * of the now trained neural network.
 */
void testing()
{
    int i,j,winner;
    int c1,c2,c3;

    c1 = 0;
    c2 = 0;
    c3 = 0;

    double max = -1;
    double err;

    FILE *correct;
    FILE *wrong;

    correct = fopen("correct.dat","w+");
    wrong   = fopen("wrong.dat","w+");

    for(i = 0; i < 3000; i++)
    {
        forward_pass(testing_set[i]);

        for(j = 0; j < p; j++)
        {
            if(network.layer[2][j].output > max)
            {
                max = network.layer[2][j].output;
                winner = j;
            }
        }

        max = -1;

        if(winner == 0 && testing_set[i].encoding[0] == 1) 
        {    
            c1++; 
            fprintf(correct,"%lf %lf\n",testing_set[i].x_1, testing_set[i].x_2); 
        }
        else if(winner == 1 && testing_set[i].encoding[1] == 1)
        {
            c2++;
            fprintf(correct,"%lf %lf\n",testing_set[i].x_1, testing_set[i].x_2);
        }
        else if(winner == 2 && testing_set[i].encoding[2] == 1)
        {
            c3++;
            fprintf(correct,"%lf %lf\n",testing_set[i].x_1, testing_set[i].x_2);
        }
        else 
            fprintf(wrong,"%lf %lf\n",testing_set[i].x_1, testing_set[i].x_2);
    }

    err = ( 1.0 - (c1 + c2 + c3) / (double) 3000 ) * 100.0;

    printf("%s+ ========= RESULTS ==========\n",KCYN);
    printf("%s|%s Class 1: %d / %d%s\n",KCYN,KGRN,c1,class_1_t,KCYN);
    printf("%s|%s Class 2: %d / %d%s\n",KCYN,KGRN,c2,class_2_t,KCYN);
    printf("%s|%s Class 3: %d / %d%s\n",KCYN,KGRN,c3,class_3_t,KCYN);
    printf("%s| ----------------------------\n",KCYN);
    printf("%s|%s Gen. Error: %2.2lf%%%s\n",KCYN,KRED,err,KCYN);
    printf("%s+ ============================\n",KCYN);

    fclose(correct);
    fclose(wrong);
}

/* 
 * Releases all the memory which was allocated
 * for the needs of this network.
 */
void free_memory()
{
    int h,i;

    for(h = 0; h < H; h++)

        for(i = 0; i < levels[h]; i++)
        {
            free(network.layer[h][i].weights);
            free(network.layer[h][i].thetas);
        }

    for(h = 0; h < H; h++)

        free(network.layer[h]);
}

void main() 
{

    srand(time(NULL));
    signal(SIGINT,stop_training);

    /* **************** */

    dataset_encoding();

    initialize_network();

    training_via_gradient_descent();

    testing();

    /* **************** */

    free_memory();

    system("gnuplot -p -e \"plot '/home/sovereign/Desktop/NeuralNetwork/final?/wrong.dat'\"");
    system("gnuplot -p -e \"plot '/home/sovereign/Desktop/NeuralNetwork/final?/correct.dat'\"");
    system("gnuplot -p -e \"plot '/home/sovereign/Desktop/NeuralNetwork/final?/error.dat'\"");

}