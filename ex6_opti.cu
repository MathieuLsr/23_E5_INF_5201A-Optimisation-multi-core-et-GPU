#include <stdio.h>
#include <math.h>

#define DIM_X 550
#define DIM_Y 550

// C Program to read a PGMB image
// and print its parameters
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
// Structure for storing the
// image data
typedef struct PGMImage {
    char pgmType[3];
    unsigned char** data;
    unsigned int width;
    unsigned int height;
    unsigned int maxValue;
    int* tab ; 
} PGMImage;
 
// Function to ignore any comments
// in file
void ignoreComments(FILE* fp)
{
    int ch;
    char line[100];
 
    // Ignore any blank lines
    while ((ch = fgetc(fp)) != EOF
           && isspace(ch))
        ;
 
    // Recursively ignore comments
    // in a PGM image commented lines
    // start with a '#'
    if (ch == '#') {
        fgets(line, sizeof(line), fp);
        ignoreComments(fp);
    }
    else
        fseek(fp, -1, SEEK_CUR);
}
 
// Function to open the input a PGM
// file and process it
bool openPGM(PGMImage* pgm,
             const char* filename)
{
    // Open the image file in the
    // 'read binary' mode
    FILE* pgmfile
        = fopen(filename, "rb");
 
    // If file does not exist,
    // then return
    if (pgmfile == NULL) {
        printf("File does not exist\n");
        return false;
    }
 
    ignoreComments(pgmfile);
    fscanf(pgmfile, "%s",
           pgm->pgmType);
 
    // Check for correct PGM Binary
    // file type
    if (strcmp(pgm->pgmType, "P5")) {
        fprintf(stderr,
                "Wrong file type!\n");
        exit(EXIT_FAILURE);
    }
 
    ignoreComments(pgmfile);
 
    // Read the image dimensions
    fscanf(pgmfile, "%d %d",
           &(pgm->width),
           &(pgm->height));
 
    ignoreComments(pgmfile);
 
    // Read maximum gray value
    fscanf(pgmfile, "%d", &(pgm->maxValue));
    ignoreComments(pgmfile);
 
    // Allocating memory to store
    // img info in defined struct
    pgm->data = (unsigned char**) malloc(pgm->height * sizeof(unsigned char*));
 
    // Storing the pixel info in
    // the struct
    printf("Type : %c\n", pgm->pgmType[1]) ;
    if (pgm->pgmType[1] == '5') {
        fgetc(pgmfile);
 
        for (int i = 0; i < pgm->height; i++) {
            pgm->data[i] = (unsigned char*) malloc(pgm->width * sizeof(unsigned char));
            
            // If memory allocation
            // is failed
            if (pgm->data[i] == NULL) {
                fprintf(stderr, "malloc failed\n");
                exit(1);
            }
 
            // Read the gray values and
            // write on allocated memory
            fread(pgm->data[i], sizeof(unsigned char), pgm->width, pgmfile);
            //printf("- %d %hhu\n", i, pgm->data[i][5]);

        }
    }
 
    // Close the file
    fclose(pgmfile);
 
    return true;
}

// Function to print the file details
void printImageDetails(PGMImage* pgm,
                       const char* filename)
{
    FILE* pgmfile = fopen(filename, "rb");
 
    // Retrieving the file extension
    char* ext = (char*) strrchr(filename, '.');
 
    if (!ext)
        printf("No extension found"
               "in file %s",
               filename);
    else
        printf("File format"
               "    : %s\n",
               ext + 1);
 
    printf("PGM File type  : %s\n",
           pgm->pgmType);
 
    // Print type of PGM file, in ascii
    // and binary format
    if (!strcmp(pgm->pgmType, "P2"))
        printf("PGM File Format:"
               "ASCII\n");
    else if (!strcmp(pgm->pgmType,
                     "P5"))
        printf("PGM File Format:"
               " Binary\n");
 
    printf("Width of img   : %d px\n",
           pgm->width);
    printf("Height of img  : %d px\n",
           pgm->height);
    printf("Max Gray value : %d\n",
           pgm->maxValue);
 
    // close file
    fclose(pgmfile);
}


void convertArrayInteger(PGMImage* pgm){

    pgm->tab = (int*) malloc(sizeof(int) * DIM_X * DIM_Y) ; 

    int x, y, offset ;
    for(x = 0 ; x < DIM_X ; x++){
        offset = x * DIM_X ; 
        for(y = 0 ; y < DIM_Y ; y++){
            pgm->tab[offset + y] = pgm->data[x][y] ; 
            
        }
    }

}
 

////////////////////////////////////////////////////////////////////////////////////


// these exist on the GPU side
texture<int, 2, cudaReadModeElementType> texIn;
//texture<int, 2, cudaReadModeElementType> texOut;


__global__ void kernel(int* resultArray) {

    int matVer[8] = {-1, -2, -1, 0, 0, 1, 2, 1} ; 
    int matHor[8] = {-1, 0, 1, 2, -2, 1, 0, -1} ; 

    //char* matPgmInitial = (char*) malloc(8 * sizeof(char)) ; 
    char matPgmInitial[8] ; 

    int verValue, horValue, sobelValue ; 
    
    int blockIdxX = blockIdx.x;
    int blockIdxY = blockIdx.y;
    int x = blockIdxX + blockIdxY * gridDim.x;
    int y = threadIdx.x;

    if(y == 0 || y == DIM_X -1) return ;

    for(x = 1 ; x < DIM_Y-1 ; x++){

        matPgmInitial[0] = tex2D(texIn, x - 1, y - 1); // pixel h g  -1
        matPgmInitial[1] = tex2D(texIn, x, y - 1);     // pixel g -1
        matPgmInitial[2] = tex2D(texIn, x + 1, y - 1); // pixel b g -1
        matPgmInitial[3] = tex2D(texIn, x - 1, y);     // pixel h   2
        matPgmInitial[4] = tex2D(texIn, x + 1, y);     // pixel b
        matPgmInitial[5] = tex2D(texIn, x - 1, y + 1); // pixel h d
        matPgmInitial[6] = tex2D(texIn, x, y + 1);     // pixel d
        matPgmInitial[7] = tex2D(texIn, x + 1, y + 1); // pixel b d

        verValue = 0 ; 
        horValue = 0 ; 

        for(int i = 0 ; i < 8 ; i++){

            verValue += matVer[i] * matPgmInitial[i] ; 
            horValue += matHor[i] * matPgmInitial[i] ; 

        }

        sobelValue =  (abs((int)(verValue / 8)) + abs((int)(horValue / 8))) / 2;
        if (sobelValue > 10) sobelValue = 200;
        resultArray[y*DIM_Y + x] = sobelValue ; 
    }

}


int main(int argc, char const* argv[])
{
    printf("Start...\n") ; 
    PGMImage* pgm = (PGMImage*) malloc(sizeof(PGMImage));
    const char* nameFile;
    nameFile = "550x550.pgm";
    cudaError_t cudaError ; 

    printf("\tname file : %s\n", nameFile);
    if (openPGM(pgm, nameFile)) {
        convertArrayInteger(pgm) ; 
        printImageDetails(pgm, nameFile);
    }

    cudaArray* cuArray;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();


    /*******************************************************************/
    if(cudaMallocArray(&cuArray, &desc, DIM_X, DIM_Y) != cudaSuccess) {
        printf("Erreur cudaMallocArray\n");
        return -1;
    }

    if(cudaMemcpyToArray(cuArray, 0, 0, pgm->tab, DIM_X * DIM_Y * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Erreur cudaMemcpyToArray\n");
        return -1;
    }

    cudaError = cudaBindTextureToArray(texIn, cuArray, desc);
    if (cudaError != cudaSuccess) {
        printf("Erreur dans le chargement de la texture #1: %s\n", cudaGetErrorString(cudaError));
        return -1;
    }

    /*
    // Allocate host memory for verification
    int* verificationArray = (int*)malloc(DIM_X * DIM_Y * sizeof(int));

    // Copy the texture data from device to host for verification
    cudaMemcpyFromArray(verificationArray, cuArray, 0, 0, DIM_X * DIM_Y * sizeof(int), cudaMemcpyDeviceToHost);

    // Print some values for verification
    for (int i = 0; i < 10; ++i) {
        printf("Verification[%d]: %d %d\n", i, verificationArray[i], pgm->tab[i]);
    }

    // Free the host memory
    free(verificationArray);
    return 0 ; 
    */

    /******************************************************************/
    /*
    int* dev_outSrc;
    if (cudaMalloc((void**)&dev_outSrc, DIM_X * DIM_Y * sizeof(int)) != cudaSuccess) {
        printf("Erreur cudaMalloc #2\n");
        return -1;
    }

    cudaArray* cuArrayOut;

    if (cudaMallocArray(&cuArrayOut, &desc, DIM_X, DIM_Y) != cudaSuccess) {
        printf("Erreur cudaMallocArray\n");
        return -1;
    }

    cudaError = cudaBindTextureToArray(texOut, cuArrayOut, desc);
    if (cudaError != cudaSuccess) {
        printf("Erreur dans le chargement de la texture #2: %s\n", cudaGetErrorString(cudaError));
        return -1;
    }
    */
    /*******************************************************************/

    int* resultArray ;
    
    if(cudaMalloc((void**)&resultArray, DIM_X * DIM_Y * sizeof(int)) != cudaSuccess){
        printf("Erreur cudaMemcpy #2\n") ; 
        return -1 ; 
    }

    //dim3 grille(11,50) ; 

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );
    
    
    kernel<<<1, DIM_Y>>>(resultArray) ; 
    
    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;
    
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf( "\n\nTime to generate: %3.1f ms\n", elapsedTime );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
 

    
    

    

    int* sobelFinal = (int*)calloc(DIM_X * DIM_Y, sizeof(int));
    if(cudaMemcpy(sobelFinal, resultArray, DIM_X * DIM_Y * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("Erreur cudaMemcpy #2\n") ; 
        return -1 ; 
    }
    
    
    FILE* f=fopen("sobelFinal.pgm","wb");
    fprintf(f,"P5\n %d %d\n255\n",DIM_X,DIM_Y);
    for (int x = 0; x < DIM_X; x++)
        for(int y=0 ; y < DIM_Y; y++){
            fputc(sobelFinal[x*DIM_X + y],f);
        }
    fclose(f);

    printf("\n\n") ; 
    return 0;
}


/*

1, 1 : Time to generate: 46.1 ms
1,550 : Avec malloc dans chaque kernel : Time to generate: 17.7 ms
1,550 : Sans malloc : Time to generate: 6.7 ms
(11,550), 550 : Avec les grilles de blocks : Time to generate: 6.5 ms



*/