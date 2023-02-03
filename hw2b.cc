#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <emmintrin.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>



int iters;//number of iterations
double left;//inclusive bound of the real axis
double right;//non-inclusive bound of the real axis.
double lower;//inclusive bound of the imaginary axis.
double upper;//non-inclusive bound of the imaginary axis.
int width;//number of points in the x-axis for output.
int height; //number of points in the y-axis for output. 
int *image;// save all image.
int *pixel;// pixel of image.
double x_range;// the interval that threads have to deal with in x axis.
double y_range;// the interval that threads have to deal with in y axis.
omp_lock_t omplock;//use to lock.
int y_pointer=0;// point the current working site in x direction.
int x_pointer=0;// point the current working site in y direction.
int rank;//Rank id
int size;//size of process
int rc;//MPI variable
MPI_Group WORLD_GROUP, USED_GROUP; //MPI Group
MPI_Comm USED_COMM = MPI_COMM_WORLD; //MPI Communication
int y_pro_range;//The range that each one have to run
int remaining;//The remaining data
int quotient;//The quotient for calculating each process has to process
int *pixel_collection; //Collection pixel from each process
int *displs; // Offset for receive buffer
int *recvcounts;//Record each the length that each process sends
double all_start= 0.0;//MPI time
double all_end= 0.0;//MPI time
double thread_start=0.0;//Calculate thread time
double thread_end=0.0;//Calculate thread time
int omp_thread;//Thread id



void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

//Use union to save length_squared value
union DataContent {
  //Record the current and next one
  double num[2];
  __m128d val;  };

int main(int argc, char **argv)
{
  /* detect how many CPUs are available */
  cpu_set_t cpu_set;
  sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
  printf("%d cpus available\n", CPU_COUNT(&cpu_set));
  int cpu_num=CPU_COUNT(&cpu_set);

  /* argument parsing */
  assert(argc == 9);
  const char *filename = argv[1];
  iters = strtol(argv[2], 0, 10);
  left = strtod(argv[3], 0);
  right = strtod(argv[4], 0);
  lower = strtod(argv[5], 0);
  upper = strtod(argv[6], 0);
  width = strtol(argv[7], 0, 10);
  height = strtol(argv[8], 0, 10);

  // MPI Init
  rc=MPI_Init(&argc, &argv);
  //all_start = MPI_Wtime();
  if(rc!=MPI_SUCCESS){
      printf("Error starting MPI program. Terminating.\n");
      MPI_Abort(MPI_COMM_WORLD,rc);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* handle arbitrary number of processes */
	if (height < size)
	{
		MPI_Comm_group(MPI_COMM_WORLD, &WORLD_GROUP);
		int range[1][3] = {{0, height - 1, 1}};
		MPI_Group_range_incl(WORLD_GROUP, 1, range, &USED_GROUP);
		MPI_Comm_create(MPI_COMM_WORLD, USED_GROUP, &USED_COMM);
		if (USED_COMM == MPI_COMM_NULL)
		{
			MPI_Finalize();
			return 0;
		}
		size = height;
	}

  quotient=height / size; 
  remaining=height % size;
  
  if(rank>=remaining){
      //remaining will distribute to the above node 
      y_pro_range=quotient;
      y_pointer=rank;
  }else{
      //read average+1 elements (to distribute remaining nums)
      y_pro_range=quotient+1;
      y_pointer=rank;
  }
  
  /* allocate memory for image */
  pixel = (int *)malloc(width * y_pro_range * sizeof(int));
  assert(pixel);

 //each segment that  have to be process
  y_range = (upper - lower)/height;
  x_range = (right - left)/width;

  // Create Lock
  omp_init_lock(&omplock);

  #pragma omp parallel num_threads(cpu_num)
  {
    //thread_start=omp_get_wtime();
    //local variable
    bool alldone=false;//Control variable to control while loop.
    const int range_size=50;//the const size that hope each thread can work.
    int workSize;//the real size that each thread has to work.
    int xStart;//the strating site of x 
    int yStart;//the strating site of y
    int x_loc_pointer;//local x pointer
    //Becaues we have to deal with two segment one times, use array to store it.
    int x_curr_pointer[2];// record current x pointer to current and next one.
    double x0_store[2];//record current x0 and next x0, the type is double to fix _mm_load
    int repeats[2];//have to record the repeat of current and next one.
    double length_squared_arr[2];// Save length_squared current and next one.
    const double constNum=2;//use to calculator
    const double constNumZero=0;//use to load
    int y_end=y_pointer+y_pro_range;//the end of range of this process.
    int thread_count=0;//for thread to count then save to image.
    double y0_re;//for handle remaining data

    while(!alldone) {
    {
      //check whether all work is done
      //the direction of working is from left to right then lower to upper.
      //Thus we can check the pointer if >=height to judge all work is done or not.
      omp_set_lock(&omplock);
      if (y_pointer >= height) {
        alldone=true;
        omp_unset_lock(&omplock);
        break;
      }
      //set the strating site to x and y
      xStart = x_pointer;
      yStart= y_pointer;
      //Record the order in this process workload 
      thread_count=int(yStart/size);

      //Every thread will get new working range when their work is done.
      //Thus we have to use lock to ensure that only a thread can get new working range at one time.
      //Set the range that threads have to work
      if(x_pointer+range_size<=width){
        workSize=range_size;
        x_pointer=x_pointer+range_size;
      }else{
        //assign the remaining range to thread then go to higher one
        workSize=width-x_pointer;
        x_pointer=0;
        //For example （7 processes）: process1 data :0,7,14,21...  process1: 1,8,15,22...
        y_pointer+=size;
        //thread_count++;
      }
      
      omp_unset_lock(&omplock);
      
      //Set y0
      __m128d y0 = _mm_set_pd1(yStart * y_range + lower);
      //For processing remaining data
      y0_re = yStart * y_range + lower;
      //Set current x pointer to current and next one
      x_curr_pointer[0]=xStart;//Current one
      x_curr_pointer[1]=xStart+1;//Next one
      //use _mm_set_pd the site will reverse
      __m128d x0 = _mm_set_pd((xStart+1)*x_range+left,xStart*x_range+left);
      //Set x0
      x0_store[0]=x_curr_pointer[0]*x_range+left;
      x0_store[1]=x_curr_pointer[1]*x_range+left;
      //process two at one time
      x_loc_pointer=xStart+2;

      //Use union to save length_squared value
      DataContent length_squared;

      //int repeats = 0;
      // double x = 0;
      // double y = 0;
      // double length_squared = 0;
      repeats[0]=0;
      repeats[1]=0;
      length_squared.val=_mm_setzero_pd();
      __m128d x=_mm_setzero_pd();
      __m128d y=_mm_setzero_pd();

      //To process the workload
      while((x_curr_pointer[0]<xStart+workSize) && (x_curr_pointer[1]<xStart+workSize)) {
        //while (repeats < iters && length_squared < 4) {
        //while((repeats[0] < iters) && (repeats[1] < iters) && (_mm_comilt_sd(length_squared.val, v_four))&&(_mm_comilt_sd(_mm_shuffle_pd(length_squared.val, length_squared.val, 1), v_four))){
        while((repeats[0] < iters) && (repeats[1] < iters) && (length_squared.num[0]<4) && (length_squared.num[1]<4)){
          //double temp = x * x - y * y + x0;
          __m128d tmp=_mm_add_pd(_mm_sub_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), x0);
          //y = 2 * x * y + y0;
          y=_mm_add_pd(_mm_mul_pd(_mm_mul_pd(x, y), _mm_load1_pd(&constNum)), y0);
          //x = temp;
          x =tmp;
          //length_squared = x * x + y * y;
          length_squared.val=_mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
          _mm_storeu_pd(length_squared_arr, length_squared.val);
          //length_squared_arr[0]=_mm_cvtsd_f64(length_squared.val);
          //repeats++;
          repeats[0]++;
          repeats[1]++;
        }
        //image[j * width + i] = repeats;
        pixel[thread_count*width+x_curr_pointer[0]] =repeats[0];
        pixel[thread_count*width+x_curr_pointer[1]] =repeats[1];
        
        //Check whether the current or next one is run out, if so perpare next one
        for(int i=0;i<2;i++){
          if (length_squared.num[i] > 4 || repeats[i] >= iters) {
              // Set next one
              x_curr_pointer[i] = x_loc_pointer++;
              x0_store[i] = x_curr_pointer[i]*x_range + left;
              repeats[i] = 0;
              //Loads a double-precision value into the low-order bits of a 128-bit vector of [2 x double]. 
              if(i==0){
                //Ex. x:(first double=the first double of constNumZero,second double=the second double of x)
                //Set x=0 for the current one and keep the nxt one
                //Reference https://zhuanlan.zhihu.com/p/572707832
                x = _mm_loadl_pd(x, &constNumZero);
                y = _mm_loadl_pd(y, &constNumZero);
                length_squared.val = _mm_loadl_pd(length_squared.val, &constNumZero);
                _mm_storeu_pd(length_squared_arr, length_squared.val);
              }else{ 
                //Loads a double-precision value into the high-order bits of a 128-bit vector of [2 x double]. 
                //Ex. x:(first double=the first double of x, second double=the second double of constNumZero)
                //Set x=0 for the next one and keep the current one
                //Reference https://zhuanlan.zhihu.com/p/572707832
                x = _mm_loadh_pd(x, &constNumZero);
                y = _mm_loadh_pd(y, &constNumZero);
                length_squared.val = _mm_loadh_pd(length_squared.val, &constNumZero);
                _mm_storeu_pd(length_squared_arr, length_squared.val);
            }
          }
        } 
        //load two double
        x0 = _mm_load_pd(x0_store);
      }
      //handle the remaining data
      for(int i=0;i<2;i++){
        if (x_curr_pointer[i] < xStart + workSize) {
          double x0 = x_curr_pointer[i]*x_range + left;
				  double x = 0;
				  double y = 0;
				  double length_squared = 0;
				  int repeats = 0;
          while (repeats < iters && length_squared < 4)
          {
            double temp = x * x - y * y + x0;
            y = 2 * x * y + y0_re;
            x = temp;
            length_squared = x * x + y * y;
            ++repeats;
          }
          //image[j * width + i] = repeats;
          pixel[thread_count * width + x_curr_pointer[i]] = repeats;
        }
      }
    }
  }
    // thread_end=omp_get_wtime();
    // omp_thread=omp_get_thread_num();
    // printf("rank:%2d,thread:%2d,Computation time:=%lf\n",rank,omp_thread,thread_end-thread_start);
  }
  //For MPI_Gatherv
  pixel_collection = (int*)malloc(width * height * sizeof(int)); 
  displs = (int*)malloc(size * sizeof(int));
  recvcounts = (int*)malloc(size * sizeof(int));

  //Set displs and recvcounts
  //i.g recvcounts[3,2,2,2] displs[0,3,5,7] process 0 start at 0 and have 3 data, process 1 start at 0 and have 5 data
  for(int i=0; i < size; i++) {
      if(i < remaining){
        recvcounts[i] = (quotient+1) * width;
      } 
      else{
        recvcounts[i] = quotient * width;
      } 
      if(i==0){
        displs[0] = 0;
      }else{
        displs[i] = displs[i-1] + recvcounts[i-1];
      } 
  }

  MPI_Gatherv(pixel, width * y_pro_range, MPI_INT,pixel_collection, recvcounts, displs, MPI_INT, 0, USED_COMM);

  if(rank == 0) {
      image = (int*)malloc(width * height * sizeof(int));
      #pragma omp parallel num_threads(cpu_num)
      {
          //pixel_collection save data like: i.g [0,1,2,0,1,0,1](diff process start from 0)
          //displs[s]: to get the ystart of each process
          //Calculate each process
          for(int s = 0; s < size; s++){
              //Use to count the number that thread do in each process because we save result form thread_count=0(yStart/size) in each process 
              int cont=0;
              for(int j = s; j < height; j+=size){
                  for(int i=0; i<width; i++){
                      //reorganization the pixel from each process
                      image[j * width + i] = pixel_collection[displs[s] + cont * width + i];
                  }
                  //save in order
                  cont++;
              }
          }
      }
      write_png(filename, iters, width, height, image);
      free(image);
  }
    free(pixel);
    // all_end = MPI_Wtime();
    MPI_Finalize();
    //  if(rank==0){
    //      double All_Time=all_end-all_start;
		// printf("All Time:%lf\n", All_Time);
	  // }
    return 0;
}



