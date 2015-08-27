
# define TAM 20000
#include "malloc.h"
#include "time.h"
#include "stdio.h"
#include "stdlib.h"
void vecAdd(int *vec1,int *vec2,int *vec3)
{
  int i;
  for ( i = 0 ; i < TAM ; i++)
  {
   vec3[i] = vec1[i] + vec2[i];   
  }  
}
// comentario random
void initialize(int *vec1,int *vec2)
{
  int i;
 	srand(time(NULL));
  for ( i = 0; i< TAM; i++)
  {
  	vec1[i] = rand() % (1+5-0) + 0;
  	vec2[i] = rand() % (1+5-0) + 0;
  }
}

void printAdd(int *a,int *b,int *c)
{
   int i;
  for ( i = 0; i < TAM ; i++)
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
  a = (int *) malloc ( sizeof(int) * TAM);
  b = (int *) malloc ( sizeof(int) * TAM);
  c = (int *) malloc ( sizeof(int) * TAM);
  initialize(a,b);
  begin = clock();
  vecAdd(a,b,c);
  end = clock();
  printAdd(a,b,c);
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Se ha demorado %f segundos.\n",time_spent);
  free(a);
  free(b);
  free(c);
}
