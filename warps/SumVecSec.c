#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void inicializador(int *vec)
{
	int contador=0;

	for(contador=0;contador <= tamano-1; contador=contador+1)
	{
		vector[contador]=rand() % 10 + 1;
	}	
}

int suma(int tamano){
	  
	clock_t begin, end;
	double time_spent;
	srand (time(NULL)); 

	int *vector=NULL;
	vector=(int *) malloc (sizeof(int)*tamano-1);



	begin = clock();

	int cont=0;
	int resultado=0;
	if (tamano % 2 != 0)
	resultado = vector[tamano-1];
	
	for(cont=0;cont < tamano-1; cont=cont+2)
	{  	
		resultado= vector[0+cont] + vector [1+cont]+resultado;
		
	}

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
			printf("Se ha demorado %f segundos.\n",time_spent);
	  
	int contador1=0;
	/*for(contador1=0;contador1 <= tamano-1; contador1=contador1+1)
	{
	printf("%d\n", vector[contador1]);
	}*/
	printf("%d\n", resultado);

	free(vector);
	return 0;
}

int main(){
  suma(200);
}

