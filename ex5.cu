#include <stdio.h>
#include <math.h>

#define N 1000

struct cuComplex {
    float r;
    float i;

    __device__ cuComplex( float a, float b ) : r(a), i(b) {}

    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia( int x, int y ) {
    
    const float scale = 0.5;
    float jx = scale * (float)(N/2 - x)/(N/2);
    float jy = scale * (float)(N/2 - y)/(N/2);
    
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);
    
    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000) return 0;
    }
    
    return 1;
}

//#define THREADS 

__global__ void kernel( unsigned char *ptr ) {

    #ifdef THREADS
    int x = threadIdx.x ;
    int y = threadIdx.y ; 
    int offset = x + y * blockDim.x;
    #else 
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    #endif

    int juliaValue = julia( x, y );
    ptr[offset] = 255 * juliaValue;
    
}

int main() {

    printf("==== Start... ====\n\n") ; 

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    unsigned char tab[N*N];
    unsigned char *gpu_tab ;

    

    for(int i=0 ; i < N*N ; i++){
        tab[i] = 0 ; 
    }
        


    if(cudaMalloc( (void**)&gpu_tab, N*N* sizeof(unsigned char) ) != cudaSuccess){
        printf("Erreur cudaMalloc #1\n") ; 
        return -1 ; 
    }
    
    if(cudaMemcpy( gpu_tab, tab, N*N * sizeof(unsigned char), cudaMemcpyHostToDevice )!= cudaSuccess){
        printf("Erreur cudaMemcpy #1\n") ; 
        return -1 ; 
    }


    dim3 grille(N,N) ; 

    cudaEventRecord( start, 0 );
    kernel<<<grille, 1>>>(gpu_tab) ; 
    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;

    if(cudaMemcpy( tab, gpu_tab, N*N * sizeof(unsigned char),cudaMemcpyDeviceToHost )!= cudaSuccess){
        printf("Erreur cudaMemcpy #2\n") ; 
        return -1 ; 
    }

    FILE* f=fopen("julia.pgm","wb");
    fprintf(f,"P5\n %d %d\n255\n",N,N);
    for (int i=0; i< N*N; i++){
        fputc(tab[i],f);
    }
    fclose(f);

    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf( "Time to generate: %3.1f ms\n", elapsedTime );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    printf("\n") ; 
    printf("==== END ====\n\n") ; 

}

/*

<<<1,1>>> : 1.5 ms 
<<<1000,1>>> : 0.2 ms
<<<1,1000>>> : 0.1 ms

*/