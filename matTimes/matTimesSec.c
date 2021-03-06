#include "malloc.h"
#include "time.h"
#include "stdio.h"
#include "stdlib.h"

int  COLA = 3;
int  ROWA = 2;
int COLB = 2;
int ROWB = COLA;

void matAdd(int *vec1,int  *vec2,int *vec3)
{
    int i, j, k;
		int aux;
    for (i = 0; i < ROWA; i++) {
        for (j = 0; j < COLB; j++) {
            aux = 0;
            for (k = 0; k < COLA; k++) {
              	aux += vec1[k + i * COLA] * vec2[j+ k * COLB];
            }
          	vec3[i*COLB + j] = aux;
        }
    }
}


void initialize(int *vec1,int *vec2)
{
  int i;
 	srand(time(NULL));
  for ( i = 0; i< COLA*ROWA; i++)
  {
  	vec1[i] = i;//rand() % (1+10-0) + 0;
  	vec2[i] = 1;//rand() % (1+20-0) + 0;
  }
}

void printTimes(int *a,int *b,int *c)
{
   int i;
  for ( i =0; i < COLB*ROWA ; i++) // ROW * ROW
  { 
  	printf("%d = %d\n",i,c[i]);
  } 
}

main () 
{
  int *a,*b,*c;
  clock_t begin, end;
	double time_spent;
	a = NULL;
  b = NULL;
  c = NULL;
  a = (int *) malloc ( sizeof(int) * COLA*ROWA);
  b = (int *) malloc ( sizeof(int) * COLB*ROWB);
  c = (int *) malloc ( sizeof(int) * ROWB*COLA);
  initialize(a,b);
  begin = clock();
  matAdd(a,b,c);
  end = clock();
  printTimes(a,b,c);
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Se ha demorado %f segundos.\n",time_spent);
  free(a);
  free(b);
  free(c);
}
