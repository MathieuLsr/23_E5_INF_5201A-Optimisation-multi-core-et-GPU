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
            //printf("- %hhu\n", pgm->data[i][5]);

        }
    }
 
    // Close the file
    fclose(pgmfile);
 
    return true;
}

void convertArrayInteger(PGMImage* pgm){

    int x, y, offset ;
    pgm->tab = (int*)calloc(pgm->width*pgm->height, sizeof(int));
    for(x = 0 ; x < pgm->width ; x++){
        offset = x * pgm->height ; 
        for(y = 0 ; y < pgm->height ; y++){
            pgm->tab[offset + y] = pgm->data[x][y] ; 
        }
    }

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

////////////////////////////////////////////////////////////////////////////////////


void kernel(PGMImage* pgm, int* sobelVer, int* sobelHor, int* sobelFinal) {


    int matVer[8] = {-1, -2, -1, 0, 0, 1, 2, 1} ; 
    int matHor[8] = {-1, 0, 1, 2, -2, 1, 0, -1} ; 

    char* matPgmInitial = (char*) malloc(8 * sizeof(char)) ; 
    int index, x, y, verValue, horValue ; 

    for(index = 0 ; index < DIM_X * DIM_Y ; index++){

        x = (int) (index / DIM_X) ;
        y = (int) (index - (x*DIM_X)) ;

        if(x == 0 || y == 0 || x == DIM_X-1 || y == DIM_Y-1) continue ; 

        matPgmInitial[0] = pgm->tab[index - 1 - DIM_Y]; //pixel h g  -1
        matPgmInitial[1] = pgm->tab[index - DIM_Y] ; //pixel g -1 
        matPgmInitial[2] = pgm->tab[index + 1 - DIM_Y] ; //pixel b g -1 

        matPgmInitial[3] = pgm->tab[index - 1];  // pixel h   2
        matPgmInitial[4] = pgm->tab[index + 1];  // pixel b

        matPgmInitial[5] = pgm->tab[index - 1 + DIM_Y + 1];  //pixel h d
        matPgmInitial[6] = pgm->tab[index + DIM_Y + 1] ;  // pixel d 
        matPgmInitial[7] = pgm->tab[index + 1 + DIM_Y + 1] ;  // pixel b d 

        verValue = 0 ; 
        horValue = 0 ; 

        for(int i = 0 ; i < 8 ; i++){

            verValue += matVer[i] * matPgmInitial[i] ; 
            horValue += matHor[i] * matPgmInitial[i] ; 

        }

        sobelVer[index] = (int)(verValue/8) ;
        sobelHor[index] = (int)(horValue/8) ;

        sobelFinal[index] = (abs((int)sobelVer[index]) + abs(sobelHor[index])) / 2 ; 

        if(sobelFinal[index] > 10) sobelFinal[index] = 200 ; 

    }

}

int main(int argc, char const* argv[])
{
    printf("Start...\n") ; 
    PGMImage* pgm = (PGMImage*) malloc(sizeof(PGMImage));
    const char* nameFile;
    nameFile = "550x550.pgm";
 
    printf("\tname file : %s\n", nameFile);
    if (openPGM(pgm, nameFile)) {
        convertArrayInteger(pgm) ; 
        printImageDetails(pgm, nameFile);
    }

    int* sobelVer = (int*)calloc(DIM_X*DIM_Y, sizeof(int));
    int* sobelHor = (int*)calloc(DIM_X*DIM_Y, sizeof(int));
    int* sobelFinal = (int*)calloc(DIM_X*DIM_Y, sizeof(int));
    
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );
    
    kernel(pgm, sobelVer, sobelHor, sobelFinal) ; 
    
    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;
    
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf( "\n\nTime to generate: %3.1f ms\n", elapsedTime );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    

    FILE* f=fopen("sobelHor.pgm","wb");
    fprintf(f,"P5\n %d %d\n255\n",DIM_X,DIM_Y);
    for (int x = 0; x < DIM_X*DIM_Y; x++)
        fputc(sobelHor[x],f);
       
    fclose(f);

    f=fopen("sobelVer.pgm","wb");
    fprintf(f,"P5\n %d %d\n255\n",DIM_X,DIM_Y);
    for (int x = 0; x < DIM_X; x++)
        for(int y=0 ; y < DIM_Y; y++){
            fputc(sobelVer[x*DIM_X + y],f);
        }
    fclose(f);

    f=fopen("sobelFinal.pgm","wb");
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

Time to generate: 46.1 ms

*/