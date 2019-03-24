/* 
 +---------------------------------------------+
 * MYE035@CSE.UOI - Computational Intelligence *
 * ------------------------------------------- *
 *  prof: Aristidis Lykas                      *
 * ------------------------------------------- *
 *  stud#1: Deligiannis Nikos          	       *
 *  stud#2: Homondozlis Paschalis 	           *
 * ------------------------------------------- *
 *  This file, generates 6000 random numbers   *
 *  with half of them inside the rectangle     *
 *  [0,2]x[0,2] while the other half is        *
 *  contained in the rectangle [0,-2]x[0,-2]   *
 +---------------------------------------------+
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void main()
{
	FILE *fp;
	int i;
	double x_1,x_2;

	fp = fopen("dataset.txt","w+");


	for (i = 0; i < 3000; i++)
	{

		x_1 = (double) rand()/RAND_MAX *  2.0; /* Random double numbers in [0,2]  */
		x_2 = (double) rand()/RAND_MAX *  2.0;

		fprintf(fp, "%f %f \n", x_1, x_2);

	}

	for (i = 0; i < 3000; i++)
    {

		x_1 = (double) rand()/RAND_MAX * -2.0; /* Random double numbers in [-2,0] */
		x_2 = (double) rand()/RAND_MAX * -2.0;
	
		fprintf(fp, "%f %f \n", x_1, x_2);
    
    }

	fclose(fp);

}
