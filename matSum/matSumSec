# define COL 5
# define ROW 10
#include "malloc.h"
#include "time.h"
#include "stdio.h"
#include "stdlib.h"

void matAdd(int *vec1,int *vec2,int *vec3)
{
  int i,j;
  for ( i = 0 ; i < COL ; i++)
  {
    for ( j = 0; j < ROW ; j++)
    {
      //printf("%d\n",i*COL+j);
   		vec3[i*COL + j] = vec1[i*COL + j] + vec2[i*COL + j];   
    } 
  }  
}


void initialize(int *vec1,int *vec2)
{
  int i;
 	srand(time(NULL));
  for ( i = 0; i< COL*ROW; i++)
  {
  	vec1[i] = rand() % (1+10-0) + 0;
  	vec2[i] = rand() % (1+20-0) + 0;
  }
}

void printAdd(int *a,int *b,int *c)
{
   int i;
  for ( i = COL*ROW/2 - 5; i < COL*ROW ; i++)
  { 
  	printf("%d + %d = %d\n",a[i],b[i],c[i]);
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
  c = (int *) malloc ( sizeof(int) * COL*ROW);
  initialize(a,b);
  begin = clock();
  matAdd(a,b,c);
  end = clock();
  printAdd(a,b,c);
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Se ha demorado %f segundos.\n",time_spent);
  free(a);
  free(b);
  free(c);
}
