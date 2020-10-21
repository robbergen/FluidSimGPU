#include "visit_writer.h"
#include <math.h>
#include <cmath>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <algorithm>
#include <future>
#include <vector>
#include "FluidGPU-unidyn.cuh"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>




const int nspts = 20100; //number of solid particles
const int nbpts = 400;//1000; //number of solid particles
const int tpts = 101;

//Storage for output
int vardims3[] = { 1,1 };
int morton_host(unsigned int x, unsigned int y, unsigned int z) {
	//int x = (bidx / GRIDSIZE / GRIDSIZE);
	//int y = (bidx / GRIDSIZE % GRIDSIZE);
	//int z = (bidx % GRIDSIZE);

	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;

	y = (y | (y << 16)) & 0x030000FF;
	y = (y | (y << 8)) & 0x0300F00F;
	y = (y | (y << 4)) & 0x030C30C3;
	y = (y | (y << 2)) & 0x09249249;

	z = (z | (z << 16)) & 0x030000FF;
	z = (z | (z << 8)) & 0x0300F00F;
	z = (z | (z << 4)) & 0x030C30C3;
	z = (z | (z << 2)) & 0x09249249;

	return x | (y << 1) | (z << 2);

}

int demorton_host(unsigned int x, int b) {
	//b should be 0 for x, 1 for y, 2 for z
	switch (b) {
	case 0: break;
	case 1: x = (x >> 1);
		break;
	case 2: x = (x >> 2);
		break;
	}
	x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x = (x | (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x | (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x | (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x | (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
}

int main(int argc, char **argv)
{
	/*
	std::cout << morton(30, 30, 30) << "\n";
	for (int k = -1; k < 2; k++)
		for (int j = -1; j < 2; j++)
			for (int i = -1; i < 2; i++)
				std::cout << morton(demorton(morton(30 + i, 30 + j, 30 + k), 0), demorton(morton(30 + i, 30 + j, 30 + k), 1), demorton(morton(30 + i, 30 + j, 30 + k), 2)) << "\n";
				*/
    //cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    int dev;

	size_t size = (nspts+nbpts) * sizeof(Particle);
	//Particle *SPptr[deviceCount];
	Particle *d_SPptr[deviceCount];
	std::vector<Particle> SPptr[2];
	Particle *d_Pbuff[deviceCount];
	int *d_cbuff[deviceCount];
    
    for (dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(dev);
        //SPptr[dev] = (Particle *)malloc(size); // Allocate particles on host
        SPptr[dev].reserve(nspts+nbpts);
		cudaMalloc((void **)&d_SPptr[dev], size); // Allocate particles on device
		cudaMalloc((void **)&d_Pbuff[dev], (nspts+nbpts)*sizeof(Particle)); // Allocate particles on device
    }

   
    float *spts[deviceCount];
    float *a3[deviceCount];
    float *b3[deviceCount];
    for (dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(dev);
        cudaMallocManaged(&spts[dev], 3*(nspts + nbpts) * sizeof(float));
        cudaMallocManaged(&a3[dev], (nspts + nbpts) * sizeof(float));
        cudaMallocManaged(&b3[dev], (nspts + nbpts) * sizeof(float));
    

	    for (int i = 0; i < 3*(nspts + nbpts); i++){
            spts[dev][i] =0 ;
        }
    }



	const char * const varnames3[] = { "subindex", "cellnumber" };
	float *arraysGPU1[] = { (float*)a3[0], (float*)b3[0], };  //only use one GPU for writing for now
    float *arraysGPU2[] = { (float*)a3[1], (float*)b3[1], };  //only use one GPU for writing for now
  

	//Set up Solid Particles
    for (dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(dev);
	    for (int j = 0; j < nspts; j++) {
		    	SPptr[dev].push_back(Particle(-.6 + 0.04*((j / 20) % 20), -0.6 + 0.04*(j % 20), -0.80 +(j / 20 /20)*0.04, 0., 0., 0.));
		    	SPptr[dev][j].index = j;
		    	SPptr[dev][j].solid = 0;
		    	SPptr[dev][j].fluid = 1;
		    	SPptr[dev][j].dens = RHO_0;
		    	SPptr[dev][j].cellnumber = int((SPptr[dev][j].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[dev][j].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[dev][j].zcoord - ZMIN) /CELLSIZE);
		    	//SPptr[j].cellnumber = morton_host(int((SPptr[j].xcoord - XMIN) / CELLSIZE), int((SPptr[j].ycoord - YMIN) / CELLSIZE), int((SPptr[j].zcoord - ZMIN)/CELLSIZE));

			}
/*
	    for (int j = 29200; j < nspts; j++) {
	    	SPptr[dev].push_back(Particle(-.5+0.05*(((j- 29200) / 10) % 10), -.5 + 0.05*((j - 29200) % 10), 0.75 + ((j - 29200) / 100)*0.05, 0., 0., 0.));
	    	SPptr[dev][j].index = j;
	    	SPptr[dev][j].solid = 0;
	    	SPptr[dev][j].fluid = 1;
	    	SPptr[dev][j].dens = RHO_0;
	    	SPptr[dev][j].cellnumber = int((SPptr[dev][j].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[dev][j].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[dev][j].zcoord - ZMIN) / CELLSIZE);
		    //SPptr[j].cellnumber = morton_host(int((SPptr[j].xcoord - XMIN) / CELLSIZE), int((SPptr[j].ycoord - YMIN) / CELLSIZE), int((SPptr[j].zcoord - ZMIN)/CELLSIZE));
	    }
*/
	    //Set up boundary particles
	    for (int i = 0; i < nbpts; i++) {
		    SPptr[dev].push_back(Particle(-0.6 + 0.06*(i % 20), -0.6 + 0.06*(i / 20), -0.96, true));
	    	SPptr[dev][nspts + i].index = nspts+i;
	    	SPptr[dev][nspts + i].solid = 1;
	    	SPptr[dev][nspts + i].fluid = 0;
	    	SPptr[dev][nspts + i].dens = RHO_0_SAND;
	    	SPptr[dev][nspts + i].cellnumber = int((SPptr[dev][i + nspts].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[dev][i + nspts].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[dev][i + nspts].zcoord - ZMIN) / CELLSIZE);
			//SPptr[i+nspts].cellnumber = morton_host(int((SPptr[i + nspts].xcoord - XMIN) / CELLSIZE), int((SPptr[i + nspts].ycoord - YMIN) / CELLSIZE), int((SPptr[i + nspts].zcoord - ZMIN) / CELLSIZE));

	    }
/*
	    for (int i = 0; i < nbpts / 8; i++) {
	    	SPptr[dev].push_back(Particle(-0.96 + 0.06*(i % 30), -0.96, -0.94 + 0.06*(i / 30), true));
	    	SPptr[dev][nspts + i + nbpts / 2].index = nspts + i + nbpts / 2;
	    	SPptr[dev][nspts + i + nbpts / 2].solid = 1;
	    	SPptr[dev][nspts + i + nbpts / 2].fluid = 0;
	    	SPptr[dev][nspts + i + nbpts / 2].dens = RHO_0_SAND;
	    	SPptr[dev][nspts + i + nbpts / 2].cellnumber = int((SPptr[dev][nspts + i + nbpts / 2].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[dev][nspts + i + nbpts / 2].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[dev][nspts + i + nbpts / 2].zcoord - ZMIN) / CELLSIZE);
		}
		for (int i = 0; i < nbpts / 8; i++) {
	    	SPptr[dev].push_back(Particle(-0.96 + 0.06*(i % 30), 0.84, -0.94 + 0.06*(i / 30), true));
	    	SPptr[dev][nspts + i + 5*nbpts / 8].index = nspts + i + 5 * nbpts / 8;
	    	SPptr[dev][nspts + i + 5*nbpts / 8].solid = 1;
	    	SPptr[dev][nspts + i + 5*nbpts / 8].fluid = 0;
	    	SPptr[dev][nspts + i + 5*nbpts / 8].dens = RHO_0_SAND;
	    	SPptr[dev][nspts + i + 5*nbpts / 8].cellnumber = int((SPptr[dev][nspts + i + 5 * nbpts / 8].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[dev][nspts + i + 5 * nbpts / 8].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[dev][nspts + i + 5 * nbpts / 8].zcoord - ZMIN) / CELLSIZE);
		}
		for (int i = 0; i < nbpts / 8; i++) {
	    	SPptr[dev].push_back(Particle(-0.96, -0.96 + 0.06*(i % 30), -0.94 + 0.06*(i / 30), true));
	    	SPptr[dev][nspts + i + 6*nbpts / 8].index = nspts + i + 6 * nbpts / 8;
	    	SPptr[dev][nspts + i + 6*nbpts / 8].solid = 1;
	    	SPptr[dev][nspts + i + 6*nbpts / 8].fluid = 0;
	    	SPptr[dev][nspts + i + 6*nbpts / 8].dens = RHO_0_SAND;
	    	SPptr[dev][nspts + i + 6*nbpts / 8].cellnumber = int((SPptr[dev][nspts + i + 6 * nbpts / 8].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[dev][nspts + i + 6 * nbpts / 8].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[dev][nspts + i + 6 * nbpts / 8].zcoord - ZMIN) / CELLSIZE);
		}
		for (int i = 0; i < nbpts / 8; i++) {
    		SPptr[dev].push_back(Particle(0.78, -0.96 + 0.06*(i % 30), -0.94 + 0.06*(i / 30), true));
    		SPptr[dev][nspts + i + 7*nbpts / 8].index = nspts + i + 7 * nbpts / 8;
    		SPptr[dev][nspts + i + 7*nbpts / 8].solid = 1;
    		SPptr[dev][nspts + i + 7*nbpts / 8].fluid = 0;
    		SPptr[dev][nspts + i + 7*nbpts / 8].dens = RHO_0_SAND;
    		SPptr[dev][nspts + i + 7*nbpts / 8].cellnumber = int((SPptr[dev][nspts + i + 7 * nbpts / 8].xcoord - XMIN) / CELLSIZE)*GRIDSIZE*GRIDSIZE + int((SPptr[dev][nspts + i + 7 * nbpts / 8].ycoord - YMIN) / CELLSIZE)*GRIDSIZE + int((SPptr[dev][nspts + i + 7 * nbpts / 8].zcoord - ZMIN) / CELLSIZE);
		
	    	//SPptr[i+nspts].cellnumber = morton_host(int((SPptr[i + nspts].xcoord - XMIN) / CELLSIZE), int((SPptr[i + nspts].ycoord - YMIN) / CELLSIZE), int((SPptr[i + nspts].zcoord - ZMIN) / CELLSIZE));
        }*/
	}
	int buffer = GRIDSIZE*GRIDSIZE; 
	//const int Ns[2] = {(NUMCELLS)/2, NUMCELLS - (NUMCELLS)/2 };
	
	//test single gpu
	const int Ns[2] = {(NUMCELLS), 0 };
	deviceCount = 1;
	buffer = 0;
	
	
	for (dev = 0; dev< deviceCount;dev++){
		if (deviceCount>1){
        for (int j = 0; j < SPptr[dev].size(); j++) {
            if (SPptr[dev][j].cellnumber < Ns[0] -buffer && dev == 1){
                SPptr[dev].erase(SPptr[dev].begin() + j);
                j--;
            }
            if (SPptr[dev][j].cellnumber >= Ns[0] +buffer && dev == 0){
                SPptr[dev].erase(SPptr[dev].begin() + j);
                j--;
            }
            //std::cout << "Cellnumber = " << SPptr[dev][j].cellnumber << ", dev = " << dev << "\n";
		}
	}
        cudaMemcpy(d_SPptr[dev], &SPptr[dev][0], SPptr[dev].size() * sizeof(Particle), cudaMemcpyHostToDevice);
	}
	

	///////Sort particles by cell number and keep track of when a new cell starts//////////
	std::vector<int> v_h[deviceCount];
	std::vector<int> particleindex[deviceCount]; 
	int *v_d[deviceCount];
	int *d_particleindex[deviceCount];
	const int N = nspts + nbpts;  // Number of elements in arrays
	size_t sizes = N * sizeof(int);
    
    for (dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(dev);
		v_h[dev].reserve(N);        // Allocate array on host
		particleindex[dev].reserve(N);        // Allocate array on host
		cudaMalloc((void **)&v_d[dev], int(SPptr[dev].size()) * sizeof(int));// Allocate array on device
		cudaMalloc((void **)&d_particleindex[dev], int(SPptr[dev].size()) * sizeof(int));// Allocate array on device
	    for (int i = 0; i<SPptr[dev].size(); i++)
	    {
			v_h[dev].push_back(SPptr[dev][i].cellnumber);
			particleindex[dev].push_back(i);
	    }

	cudaMemcpy(v_d[dev], &v_h[dev][0], int(SPptr[dev].size()) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_particleindex[dev], &particleindex[dev][0], int(SPptr[dev].size()) * sizeof(int), cudaMemcpyHostToDevice);
    //std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";
	}
	
	int *xleft[deviceCount];// = {0};
    int *xright[deviceCount];//= {0};
    int *sizeleft[deviceCount];//= {0};
    int *sizeright[deviceCount];//= {0};
    int *d_xl[deviceCount];
    int *d_xr[deviceCount];
    int *d_sl[deviceCount];
    int *d_sr[deviceCount];
    for(int dev=0; dev<deviceCount; dev++) {
        cudaSetDevice(dev);
        xleft[dev] = (int *)malloc(sizeof(int));
        xright[dev] = (int *)malloc(sizeof(int));
        sizeleft[dev] = (int *)malloc(sizeof(int));
        sizeright[dev] = (int *)malloc(sizeof(int));
        cudaMalloc((void **)&d_xl[dev], sizeof(int));// Allocate array on device
        cudaMalloc((void **)&d_xr[dev], sizeof(int));// Allocate array on device
        cudaMalloc((void **)&d_sl[dev], sizeof(int));// Allocate array on device
        cudaMalloc((void **)&d_sr[dev], sizeof(int));// Allocate array on device
    }
/*
    thrust::device_ptr<Particle> t_a[2];
    thrust::device_ptr<int> t_v[2];
    for (dev = 0; dev < deviceCount; dev++){
        cudaSetDevice(dev);
        thrust::device_ptr<Particle> t_b(d_SPptr[dev]);
        thrust::device_ptr<int> t_x(v_d[dev]);
        t_a[dev] = t_b;
        t_v[dev] = t_x;

    }
*/
    
	int *start, *end, *split, *numsplit, *d_start[deviceCount], *d_end[deviceCount],*d_split[deviceCount], *d_numsplit, *start_copy, *d_start_copy[deviceCount];
    size_t sizes2[2] = {(Ns[0] +buffer)* sizeof(int),(Ns[1]+buffer)*sizeof(int)};

	start = (int *)malloc(NUMCELLS*sizeof(int));        // Allocate array on host
	start_copy = (int *)malloc(NUMCELLS*sizeof(int));        // Allocate array on host
	end = (int *)malloc(NUMCELLS*sizeof(int));        // Allocate array on host
	split = (int *)malloc(NUMCELLS*sizeof(int));        // Allocate array on host
	numsplit = (int *)malloc(sizeof(int));        // Allocate array on host
    numsplit[0] = 0;

    for (int i = 0; i<NUMCELLS; i++)
    {
		start[i] = -1;
		start_copy[i] = -1;
		end[i] = -1;
		split[i] = -1;
    }

    
   
    for(int dev=0, pos=0; dev<deviceCount; pos+=Ns[dev], dev++) {
        cudaSetDevice(dev);
        cudaMalloc((void **)&d_start[dev], sizes2[dev]);// Allocate array on device
		//std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";
		cudaMalloc((void **)&d_start_copy[dev], sizes2[dev]);// Allocate array on device
        //std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";
        cudaMalloc((void **)&d_end[dev], sizes2[dev]);// Allocate array on device
		//std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";
		cudaMalloc((void **)&d_split[dev], sizes2[dev]);// Allocate array on device
		//std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";
		cudaMalloc((void **)&d_numsplit, sizeof(int));// Allocate array on device
        //std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";

        cudaMemcpy(d_start[dev], start+pos-(dev>0)*buffer, sizes2[dev], cudaMemcpyHostToDevice);
		//std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";
		cudaMemcpy(d_start_copy[dev], start_copy+pos-(dev>0)*buffer, sizes2[dev], cudaMemcpyHostToDevice);
        //std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";
        cudaMemcpy(d_end[dev], end+pos-(dev>0)*buffer, sizes2[dev], cudaMemcpyHostToDevice);
		//std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";
		cudaMemcpy(d_split[dev], split+pos-(dev>0)*buffer, sizes2[dev], cudaMemcpyHostToDevice);
		//std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";
		cudaMemcpy(d_numsplit, numsplit, sizeof(int), cudaMemcpyHostToDevice);
        //std::cout << cudaGetErrorName(cudaGetLastError())<< "\n";
    }

	int dsz[2] ={int(SPptr[0].size()),int(SPptr[1].size())}; 
	for (int t = 0; t < tpts; t++) {
		std::cout << "t= " << t << "\n";
		//std::cout << SPptr[0].size() << " particles on device 0 at t="<< t<< "\n";
		//std::cout << SPptr[1].size() << " particles on device 1 at t="<< t<< "\n";
		
		cudaEvent_t start, stop;
		CUDA_CHECK_RETURN(cudaEventCreate(&start));
		CUDA_CHECK_RETURN(cudaEventCreate(&stop));
		float elapsedTime;
		CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
		cudaDeviceSynchronize();
        for (dev = 0; dev < deviceCount; dev++){
			cudaSetDevice(dev);
			
			//thrust::device_ptr<Particle> t_b(d_SPptr[dev]);
			thrust::device_ptr<int> t_b(d_particleindex[dev]);
			thrust::device_ptr<int> t_x(v_d[dev]);

			thrust::sort_by_key(t_x, t_x + dsz[dev], t_b);
		   
			if (strcmp(cudaGetErrorName(cudaGetLastError()),"cudaSuccess")!=0){
			//	std::cout << "Sorting failed at t = "<< t << ", " << cudaGetErrorName(cudaGetLastError())<< "\n";
			}
		}


		
		for (dev = 0; dev < deviceCount; dev++){
			cudaSetDevice(dev);
            findneighbours << <NUMCELLS, 1024 >> > (v_d[dev], d_start[dev], d_start_copy[dev], d_end[dev],dsz[dev], dev*(Ns[0]-buffer)); //(sorted list of particles, start cells, end cells, number of particles, start index)
			if (strcmp(cudaGetErrorName(cudaGetLastError()),"cudaSuccess")!=0){
			//	std::cout << "Binning failed at t = "<< t << ", " << cudaGetErrorName(cudaGetLastError())<< "\n";
			}
		}

		
		for (dev = 0; dev < deviceCount; dev++){
			cudaSetDevice(dev);
			mykernel <<<NUMCELLS, 1024 >> > (d_SPptr[dev], d_particleindex[dev], v_d[dev], d_start[dev], d_end[dev], d_split[dev], dsz[dev],Ns[dev]+buffer,dev,buffer,d_numsplit);
			//cudaDeviceSynchronize();
			if (strcmp(cudaGetErrorName(cudaGetLastError()),"cudaSuccess")!=0){
				std::cout << "Force calc failed at t = "<< t << ",  " << cudaGetErrorName(cudaGetLastError())<< "\n";
			}
		}
			cudaMemcpy(numsplit, d_numsplit, sizeof(int), cudaMemcpyDeviceToHost);
			

			//std::cout<<numsplit[0] << "\n";
			//sort start by split
			thrust::device_ptr<int> t_1(d_start_copy[0]);
			thrust::device_ptr<int> t_2(d_split[0]);
			thrust::sort_by_key(t_2, t_2 + NUMCELLS, t_1,  thrust::greater<int>());

			for (dev = 0; dev < deviceCount; dev++){
				cudaSetDevice(dev);
				mykernel3 <<<numsplit[0]*8, 240 >> > (d_SPptr[dev], d_particleindex[dev], v_d[dev], d_start[dev], d_end[dev], d_split[dev], dsz[dev],Ns[dev]+buffer,dev,buffer,d_numsplit);
				cudaDeviceSynchronize();
				if (strcmp(cudaGetErrorName(cudaGetLastError()),"cudaSuccess")!=0){
					std::cout << "Force calc #2 failed at t = "<< t << ",  " << cudaGetErrorName(cudaGetLastError())<< "\n";
				}
			}


		
		for (dev = 0; dev < deviceCount; dev++){
			cudaSetDevice(dev);
			mykernel2 << <NUMCELLS, 1024 >> > (d_SPptr[dev], d_particleindex[dev], v_d[dev], d_start_copy[dev], d_start[dev], d_end[dev],d_split[dev],d_numsplit,dsz[dev], Ns[dev]+buffer, dev, buffer, spts[dev], a3[dev], b3[dev]);
			if (strcmp(cudaGetErrorName(cudaGetLastError()),"cudaSuccess")!=0){
			//	std::cout << "Updating failed at t = "<< t << ",  " << cudaGetErrorName(cudaGetLastError())<< "\n";
			}
		}

		if (deviceCount>1){
		for (dev = 0; dev < deviceCount; dev++){
            cudaSetDevice(dev);
			find_idx << <NUMCELLS, 1024 >> > (v_d[dev], dev, dsz[dev], buffer, d_xl[dev], d_xr[dev], d_sl[dev], d_sr[dev]);
			if (strcmp(cudaGetErrorName(cudaGetLastError()),"cudaSuccess")!=0){
				std::cout << "Index search failed at t = "<< t << ",  " << cudaGetErrorName(cudaGetLastError())<< "\n";
            }
            cudaDeviceSynchronize();
            cudaMemcpy(&xleft[dev][0], d_xl[dev],sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&xright[dev][0], d_xr[dev],sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sizeleft[dev][0], d_sl[dev],sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sizeright[dev][0], d_sr[dev],sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
		}

		for (dev = 0; dev<deviceCount; dev++){
			dsz[dev] = dev==0 ? (xright[dev][0]-xleft[dev][0]+1) + (sizeleft[(dev+1)][0]-xleft[(dev+1)][0]) : (xright[dev-1][0]-sizeright[dev-1][0]+1) +  (xright[dev][0]-xleft[dev][0]+1);  
		}

		//take sleft-xleft particles from dev x at xleft to dev x-1 at xright+1 of devx-1
		//new max index of dev x-1 is xright_{x-1} + (sleft-xleft)_{x} 

	//use particleindex here later
		dev= 1;
		cudaSetDevice(1);
		cudaMemcpy(&SPptr[dev-1][xright[dev-1][0]+1],d_SPptr[dev]+xleft[dev][0],(sizeleft[dev][0]-xleft[dev][0])*sizeof(Particle), cudaMemcpyDeviceToHost);
		cudaSetDevice(0);
		cudaMemcpy(d_SPptr[dev-1]+xright[dev-1][0]+1,&SPptr[dev-1][xright[dev-1][0]+1],(sizeleft[dev][0]-xleft[dev][0])*sizeof(Particle), cudaMemcpyHostToDevice);

		//how many particles will be transferred to dev x = (xright-sright+1)_{x-1}
		//therefore need to shift dev x particles to the left by xleft_{x}-(xright-sright+1)_{x-1}
		//shift particle indices xleft_x --> xright_{x} + (sleft-xleft)_{x+1} 

//use particleindex here later
        cudaSetDevice(1);
        mem_shift << <NUMCELLS, 1024 >> > (d_SPptr[dev], d_Pbuff[dev], v_d[dev], d_cbuff[dev], dev,xleft[dev][0]-(xright[dev-1][0]-sizeright[dev-1][0]+1),xleft[dev][0],xright[dev][0]);
        cudaDeviceSynchronize();
        if (strcmp(cudaGetErrorName(cudaGetLastError()),"cudaSuccess")!=0){
			std::cout << "Mem shift failed at t = "<< t << ",  " << cudaGetErrorName(cudaGetLastError())<< "\n";
        }
            

		
		//now that particles shifted, indices shift as well
		int xleftold[2];
		int xrightold[2];
		int sizeleftold[2];
		int sizerightold[2];
		for (dev = 0; dev<deviceCount; dev++){
			xleftold[dev] = xleft[dev][0];
			xrightold[dev] = xright[dev][0];
			sizeleftold[dev] = sizeleft[dev][0];
			sizerightold[dev] = sizeright[dev][0];
		}
		dev = 1;
		xleft[dev][0] -= xleftold[dev]-(xrightold[dev-1]-sizerightold[dev-1]+1);
		xright[dev][0] -= xleftold[dev]-(xrightold[dev-1]-sizerightold[dev-1]+1);
		sizeleft[dev][0] -= xleftold[dev]-(xrightold[dev-1]-sizerightold[dev-1]+1);
		sizeright[dev][0] -= xleftold[dev]-(xrightold[dev-1]-sizerightold[dev-1]+1);
			
//use particleindex here later
		//take xrightold-srightold+1 particles from dev x at srightnew to dev x+1 at 0
        dev = 0;
		cudaSetDevice(dev);
		cudaMemcpy(&SPptr[dev+1][0],d_SPptr[dev]+sizeright[dev][0],(xrightold[dev]-sizerightold[dev]+1)*sizeof(Particle), cudaMemcpyDeviceToHost);
		cudaSetDevice(dev+1);
		cudaMemcpy(d_SPptr[dev+1],&SPptr[dev+1][0],(xrightold[dev]-sizerightold[dev]+1)*sizeof(Particle), cudaMemcpyHostToDevice);
		


		for (dev = 0;dev<deviceCount;dev++){
			dsz[dev] = dev==0 ? (xright[dev][0]-xleft[dev][0]+1) + (sizeleft[(dev+1)][0]-xleft[(dev+1)][0]) : (xright[dev-1][0]-sizeright[dev-1][0]+1) +  (xright[dev][0]-xleft[dev][0]+1);
		}
		
	}

		if (t % 30 == 0) {
			for (dev = 0; dev < deviceCount; dev++){
				cudaSetDevice(dev);
				cudaMemcpy(&SPptr[dev][0], d_SPptr[dev], dsz[dev] * sizeof(Particle), cudaMemcpyDeviceToHost); //copy updated particles back to cpu
				if (strcmp(cudaGetErrorName(cudaGetLastError()),"cudaSuccess")!=0){
					std::cout << "Particle copy failed at t = "<< t << ",  " << cudaGetErrorName(cudaGetLastError())<< "\n";
				}
			}
			cudaDeviceSynchronize();
			//Write each frame to file
			std::ostringstream oss;
			std::ostringstream oss2;
			oss << "anim-uni/anim_s_GPU0_" << t / 1<< ".vtk";
			std::string var = oss.str();
			const char* cstr = var.c_str();
			write_point_mesh(cstr, 0, dsz[0], spts[0], 2, vardims3, varnames3, arraysGPU1);
			oss2 << "anim-uni/anim_s_GPU1_" << t / 1 << ".vtk";
			std::string var2 = oss2.str();
			const char* cstr2 = var2.c_str();
			//write_point_mesh(cstr2, 0, dsz[1], spts[1], 2, vardims3, varnames3, arraysGPU2);

		}
		//Recalculate cells
		for (dev = 0; dev< deviceCount; dev++){ //take xright-sright particles from dev x at sright to dev x+1 at 0
			cudaSetDevice(dev);
			//recalculate sizes of arrays
			cell_calc << <NUMCELLS, 1024 >> >(d_SPptr[dev], d_particleindex[dev], v_d[dev], dsz[dev],dev);
			cudaDeviceSynchronize();
		}



		cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));

		CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
		CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
		CUDA_CHECK_RETURN(cudaGetLastError());
		CUDA_CHECK_RETURN(cudaEventDestroy(start));
		CUDA_CHECK_RETURN(cudaEventDestroy(stop));
		std::cout << "\nElapsed kernel time: " << elapsedTime << " ms\n";


		//std::cout << cudaGetErrorName(cudaGetLastError()) << "\n";
		

//		cudaDeviceSynchronize();

		
	
	}
	return 0;
}
