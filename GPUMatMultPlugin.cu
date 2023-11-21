#include "GPUMatMultPlugin.h"


void GPUMatMultPlugin::input(std::string infile) {
  readParameterFile(infile);
}

void GPUMatMultPlugin::run() {}

void GPUMatMultPlugin::output(std::string outfile) {
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;
int M, N, P;
 M = atoi(myParameters["M"].c_str());
 N = atoi(myParameters["N"].c_str());
 P = atoi(myParameters["P"].c_str());
 numARows = M;
 numAColumns = N;
 numBRows = N;
 numBColumns = P;
 numCRows = M;
 numCColumns = P;

  hostA = (float*) malloc (M*N*sizeof(float));
  hostB = (float*) malloc (N*P*sizeof(float));
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numARows * numBColumns * sizeof(float));

  numCRows    = numARows;
  numCColumns = numBColumns;

 std::ifstream myinput((std::string(PluginManager::prefix())+myParameters["matrix1"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < M*N; ++i) {
	float k;
	myinput >> k;
        hostA[i] = k;
 }
 std::ifstream myinput2((std::string(PluginManager::prefix())+myParameters["matrix2"]).c_str(), std::ios::in);
 for (i = 0; i < N*P; ++i) {
	float k;
	myinput2 >> k;
        hostB[i] = k;
 }

  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceA,
                     numARows * numAColumns * sizeof(float));
  cudaMalloc((void **)&deviceB,
                     numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **)&deviceC,
                     numARows * numBColumns * sizeof(float));
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA,
                     numARows * numAColumns * sizeof(float),
                     cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB,
                     numBRows * numBColumns * sizeof(float),
                     cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(16, 16);
// changed to BColumns and ARows from Acolumns and BRows
  dim3 gridDim(ceil(((float)numBColumns) / blockDim.x),
               ceil(((float)numARows) / blockDim.y));

  //@@ Launch the GPU Kernel here
  sgemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, numARows,
                               numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  cudaMemcpy(hostC, deviceC,
                     numARows * numBColumns * sizeof(float),
                     cudaMemcpyDeviceToHost);
	std::ofstream outsfile(outfile.c_str(), std::ios::out);

        for (i = 0; i < M*P; ++i){
		outsfile << hostC[i];//std::setprecision(0) << a[i*N+j];
		outsfile << "\n";
	}
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  free(hostA);
  free(hostB);
  free(hostC);

}
PluginProxy<GPUMatMultPlugin> GPUMatMultPluginProxy = PluginProxy<GPUMatMultPlugin>("GPUMatMult", PluginManager::getInstance());


