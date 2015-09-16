# define COL 3
# define ROW 2
#include "malloc.h"
#include "time.h"
#include "stdio.h"
#include "stdlib.h"

void matAdd(int *vec1,int *vec2,int *vec3)
{
    int i, j, k;

    for (i = 0; i < ROW; i++) {
        for (j = 0; j < COL; j++) {
            vec3[j + i * COL] = 0;
            for (k = 0; k < ROW; k++) {
                vec3[j + i * COL] += vec1[k + i * ROW] * vec2[j + k * COL];
            }
        }
    }
}


void initialize(int *vec1,int *vec2)
{
  int i;
 	srand(time(NULL));
  for ( i = 0; i< COL*ROW; i++)
  {
  	vec1[i] = i;//rand() % (1+10-0) + 0;
  	vec2[i] = 1;//rand() % (1+20-0) + 0;
  }
}

void printTimes(int *a,int *b,int *c)
{
   int i;
  for ( i =0; i < ROW*ROW ; i++)
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
  a = (int *) malloc ( sizeof(int) * COL*ROW);
  b = (int *) malloc ( sizeof(int) * COL*ROW);
  c = (int *) malloc ( sizeof(int) * ROW*ROW);
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
