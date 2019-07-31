/*
This is the tool ....

Contact Author: Jian Tang, Microsoft Research, jiatang@microsoft.com, tangjianpku@gmail.com
Publication: Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Qiaozhu Mei. "LINE: Large-scale Information Network Embedding". In WWW 2015.
*/

// Format of the training file:
//
// The training file contains serveral lines, each line represents a DIRECTED edge in the network.
// More specifically, each line has the following format "<u> <v> <w>", meaning an edge from <u> to <v> with weight as <w>.
// <u> <v> and <w> are seperated by ' ' or '\t' (blank or tab)
// For UNDIRECTED edge, the user should use two DIRECTED edges to represent it.


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>
#include <gsl/gsl_rng.h>


#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

const int hash_table_size = 30000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

typedef float real;                    // Precision of float numbers

struct ClassVertex {
	double degree;
	char *name;
};

char network_file[MAX_STRING], embedding_file[MAX_STRING];
struct ClassVertex *vertex;
int is_binary = 0, thread_count = 1, order = 2, dim = 100, num_negative = 5;
int *vertex_hash_table, *neg_table;
int max_num_vertices = 1000, num_vertices = 0;
long long total_samples = 1, current_sample_count = 0, num_edges = 0;
real init_rho = 0.025, rho;
real *emb_vertex, *emb_context, *sigmoid_table;

int *edge_source_id, *edge_target_id;
double *edge_weight;

// Parameters for edge sampling
long long *alias;
double *prob;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

/* Build a hash table, mapping each vertex name to a unique vertex id */
//Pretty sure it can only contain 30000000 (hash_table_size) many vertices
unsigned int Hash(char *key)
{
	unsigned int seed = 131;
	unsigned int hash = 0;
	while (*key) //Loop over each character in the string
	{
		hash = hash * seed + (*key++);
	}
	return hash % hash_table_size;
}

/*
Initialize hashtable 
Each entries gets a value of -1
*/
void InitHashTable()
{
	vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
	#pragma omp parallel for num_threads(thread_count)
	for (int k = 0; k < hash_table_size; k++){
		vertex_hash_table[k] = -1;
	}
}

//Insert the vertex into the hashtable assigning it as value the current number of vertices
void InsertHashTable(char *key, int value)
{
	int addr = Hash(key);	//Get the unique vertex id for the vertex
	while (vertex_hash_table[addr] != -1) addr = (addr + 1) % hash_table_size;
	vertex_hash_table[addr] = value;
}

//Returns the index of the vertex or -1 if it is new
int SearchHashTable(char *key)
{
	int addr = Hash(key);	//Get the unique vertex id for the vertex
	while (1)
	{
		if (vertex_hash_table[addr] == -1) return -1;	//Return -1 if the vertex is new
		if (!strcmp(key, vertex[vertex_hash_table[addr]].name)) return vertex_hash_table[addr]; //If the input vertex name and the previously for this address assigned name are equal return this adress
		addr = (addr + 1) % hash_table_size; //Else increment the adress and try it again
	}
	return -1;
}

/* Add a vertex to the vertex set */
int AddVertex(char *name)
{
	int length = strlen(name) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vertex[num_vertices].name = (char *)calloc(length, sizeof(char));
	//Assign it the name and set the degree to 0
	strncpy(vertex[num_vertices].name, name, length-1);
	vertex[num_vertices].degree = 0;
	num_vertices++;
	if (num_vertices + 2 >= max_num_vertices) //If the current vertex array only has one entry left increase its size by 1000
	{
		max_num_vertices += 1000;
		vertex = (struct ClassVertex *)realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));
	}
	InsertHashTable(name, num_vertices - 1);
	return num_vertices - 1;
}

/* Read network from the training file */
void ReadData()
{
	FILE *fin;
	char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
	int vid;
	double weight;

	fin = fopen(network_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: network file not found!\n");
		exit(1);
	}
	num_edges = 0;
	while (fgets(str, sizeof(str), fin)) num_edges++;	//Determine number of edges
	fclose(fin);
	printf("Number of edges: %lld          \n", num_edges);

	//Init memory for edge vectors
	edge_source_id = (int *)malloc(num_edges*sizeof(int));
	edge_target_id = (int *)malloc(num_edges*sizeof(int));
	edge_weight = (double *)malloc(num_edges*sizeof(double));
	if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	fin = fopen(network_file, "rb");
	num_vertices = 0;
	//Reading edges
	for (int k = 0; k != num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

		if (k % 10000 == 0)
		{
			printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
			fflush(stdout);
		}

		vid = SearchHashTable(name_v1);	//Returns the index of the vertex or -1 if it is new
		if (vid == -1) vid = AddVertex(name_v1); //Add the new vertex to the set
		vertex[vid].degree += weight; //Increment the vertex degree by the weight of the edge
		edge_source_id[k] = vid; //Set the source vertex of the edge to the vertex index

		vid = SearchHashTable(name_v2); //Returns the index of the vertex or -1 if it is new
		if (vid == -1) vid = AddVertex(name_v2); //Add the new vertex to the set
		vertex[vid].degree += weight; //Increment the vertex degree by the weight of the edge
		edge_target_id[k] = vid; //Set the source vertex of the edge to the vertex index

		edge_weight[k] = weight;
	}
	fclose(fin);
	printf("Number of vertices: %d          \n", num_vertices);
}

/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable()
{
	//Allocate arrays with size equal to the number of edges
	alias = (long long *)malloc(num_edges*sizeof(long long));
	prob = (double *)malloc(num_edges*sizeof(double));
	if (alias == NULL || prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double *norm_prob = (double*)malloc(num_edges*sizeof(double));
	long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
	long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
	if (norm_prob == NULL || large_block == NULL || small_block == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double sum = 0;
	long long cur_small_block, cur_large_block;
	long long num_small_block = 0, num_large_block = 0;

	#pragma omp parallel for num_threads(thread_count) reduction(+:sum)
	for (long long k = 0; k < num_edges; k++){ 
		sum += edge_weight[k]; //Sum up all edge weights
	}

	#pragma omp parallel for num_threads(thread_count)
	for (long long k = 0; k < num_edges; k++){
		norm_prob[k] = edge_weight[k] * num_edges / sum; //Each edge gets a probability based on its weight divided by the total weight of all edges
	}	

	for (long long k = num_edges - 1; k >= 0; k--) //Divide the edges in two blocks
	{
		if (norm_prob[k]<1) 
			small_block[num_small_block++] = k; //Assign edge index to small_block if probability is smaller than the average
		else
			large_block[num_large_block++] = k; //Assign edge index to large_block if probability is higher than the average
	}

	while (num_small_block && num_large_block)
	{
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];
		prob[cur_small_block] = norm_prob[cur_small_block];
		alias[cur_small_block] = cur_large_block;
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	while (num_large_block) prob[large_block[--num_large_block]] = 1;
	while (num_small_block) prob[small_block[--num_small_block]] = 1;

	free(norm_prob);
	free(small_block);
	free(large_block);
}

long long SampleAnEdge(double rand_value1, double rand_value2)
{
	long long k = (long long)num_edges * rand_value1;
	return rand_value2 < prob[k] ? k : alias[k];
}

/* Initialize the vertex embedding and the context embedding */
void InitVector()
{
	long long a, b;

	a = posix_memalign((void **)&emb_vertex, 128, (long long)num_vertices * dim * sizeof(real)); //For each vertex a float vector of size dim is allocated
	if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	#pragma omp parallel for num_threads(thread_count) collapse(2)
	for (b = 0; b < dim; b++){
		for (a = 0; a < num_vertices; a++){
			emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;	//Create random values in the intervall between -0.5 and 0.5 and divide them through the numbe of dimensions
		}
	}
	a = posix_memalign((void **)&emb_context, 128, (long long)num_vertices * dim * sizeof(real));
	if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	#pragma omp parallel for num_threads(thread_count) collapse(2)
	for (b = 0; b < dim; b++){
		for (a = 0; a < num_vertices; a++){
			emb_context[a * dim + b] = 0;	//Context vectors are initialized as 0 in all dimensions
		}
	}
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable()
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	neg_table = (int *)malloc(neg_table_size * sizeof(int)); //Allocate array of int with 100 million entries
	#pragma omp parallel for num_threads(thread_count) reduction(+:sum)
	for (int k = 0; k < num_vertices; k++){
		sum += pow(vertex[k].degree, NEG_SAMPLING_POWER); //Sum the vertex degrees (sum of weights of connected edges) to the power of 0.75
	}	
	for (int k = 0; k != neg_table_size; k++)
	{
		if ((double)(k + 1) / neg_table_size > por)
		{
			cur_sum += pow(vertex[vid].degree, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
		}
		neg_table[k] = vid - 1;
	}
}

/* Fastly compute sigmoid function */
//sigmoid_table_size = 1000
//SIGMOID_BOUND = 6
/*
Pre calculate the sigmoid values for k = 0 ... 1000
The result is the joint probability between two vertexes
*/
void InitSigmoidTable()
{
	real x;
	sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real)); 
	#pragma omp parallel for num_threads(thread_count)
	for (int k = 0; k < sigmoid_table_size; k++){
		x = 2.0 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
		sigmoid_table[k] = 1 / (1 + exp(-x));
	}
}

/*
Returns the sigmoid (joint probability) of the vertex product (x)
*/
real FastSigmoid(real x)
{
	if (x > SIGMOID_BOUND) return 1;
	else if (x < -SIGMOID_BOUND) return 0;
	int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
	return sigmoid_table[k];
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
	seed = seed * 25214903917 + 11;
	return (seed >> 16) % neg_table_size;
}

/* Update embeddings */
/*
Updates the embeddings of the target vertex vec_v and adds the change regarding the source vertex to vec_error
*/
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
	real x = 0, g;
	#pragma omp parallel for num_threads(thread_count) reduction(+:x)
	for (int c = 0; c <= dim; c++){
		x += vec_u[c] * vec_v[c]; //Calculate dot product of the two vectors
	}	
	g = (label - FastSigmoid(x)) * rho;
	#pragma omp parallel for num_threads(thread_count)
	for (int c = 0; c < dim; c++){
		vec_error[c] += g * vec_v[c];
	}
	#pragma omp parallel for num_threads(thread_count)
	for (int c = 0; c < dim; c++){
		vec_v[c] += g * vec_u[c];
	}
}

//Thread to train the model
//Probably not threadsafe at all
void *TrainLINEThread(void *id)
{
	long long u, v, lu, lv, target, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id; //Init seed with the thread id
	real *vec_error = (real *)calloc(dim, sizeof(real));

	while (1)
	{
		//judge for exit
		//Exit when the thread has done its share of the total number of samples to consider
		if (count > total_samples / thread_count + 2) break;

		//After 10000 samples print an update to the console and update variables
		if (count - last_count > 10000)
		{
			current_sample_count += count - last_count; //Increment the global current_sample_count by the number of samples the current thread has done since the last check
			last_count = count;
			printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100); //Print info to console
			fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

		curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r)); //Sample an edge
		u = edge_source_id[curedge];
		v = edge_target_id[curedge];

		lu = u * dim; //Get start point of embedding vector for the source vertex
		for (int c = 0; c != dim; c++) vec_error[c] = 0; //Set error vector to 0

		// NEGATIVE SAMPLING
		for (int d = 0; d != num_negative + 1; d++)
		{
			if (d == 0) // Update with real edge
			{
				target = v;
				label = 1;
			}
			else // Update with negative edge
			{
				target = neg_table[Rand(seed)];
				label = 0;
			}
			lv = target * dim; //Get start point of embedding vector for the target vertex
			if (order == 1) Update(&emb_vertex[lu], &emb_vertex[lv], vec_error, label);
			if (order == 2) Update(&emb_vertex[lu], &emb_context[lv], vec_error, label); //For second-order use the context embedding of the target vertex
		}
		for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error[c]; //Iüdate embedding vector of source vertex according to error vector

		count++;
	}
	free(vec_error);
	pthread_exit(NULL);
}

/*
Write embeddings to file
*/
void Output()
{
	FILE *fo = fopen(embedding_file, "wb");
	fprintf(fo, "%d %d\n", num_vertices, dim);
	for (int a = 0; a < num_vertices; a++)
	{
		fprintf(fo, "%s ", vertex[a].name);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
		else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}

void TrainLINE() {
	long a;
	pthread_t *pt = (pthread_t *)malloc(thread_count * sizeof(pthread_t)); //Allocate memory for the threads

	if (order != 1 && order != 2)
	{
		printf("Error: order should be either 1 or 2!\n");
		exit(1);
	}
	printf("--------------------------------\n");
	printf("Order: %d\n", order);
	printf("Samples: %lldM\n", total_samples / 1000000);
	printf("Negative: %d\n", num_negative);
	printf("Dimension: %d\n", dim);
	printf("Initial rho: %lf\n", init_rho);
	printf("--------------------------------\n");

	InitHashTable();
	ReadData();
	InitAliasTable(); //Init alias sampling to allow sampling of edges according to their probability
	InitVector(); //Init vertex embeddings
	InitNegTable();
	InitSigmoidTable(); //Precalculate the sigmoid values (joint probability)

	gsl_rng_env_setup();
	gsl_T = gsl_rng_rand48; //This is the Unix rand48 generator
	gsl_r = gsl_rng_alloc(gsl_T); //Init the generator with above type
	gsl_rng_set(gsl_r, 314159265); //Set the seed of the generator

	clock_t start = clock();
	printf("--------------------------------\n");
	for (a = 0; a < thread_count; a++) pthread_create(&pt[a], NULL, TrainLINEThread, (void *)a); //Init the vectors and each vector get its id
	for (a = 0; a < thread_count; a++) pthread_join(pt[a], NULL);
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

	Output();
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("LINE: Large Information Network Embedding\n");
		printf("Thread-Safe Version\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse network data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the learnt embeddings\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
		printf("\t-size <int>\n");
		printf("\t\tSet dimension of vertex embeddings; default is 100\n");
		printf("\t-order <int>\n");
		printf("\t\tThe type of the model; 1 for first order, 2 for second order; default is 2\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5\n");
		printf("\t-samples <int>\n");
		printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-rho <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\nExamples:\n");
		printf("./line -train net.txt -output vec.txt -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
		return 0;
	}
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-order", argc, argv)) > 0) order = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) thread_count = atoi(argv[i + 1]);
	total_samples *= 1000000;
	rho = init_rho;
	vertex = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex)); //Init vertex array with memory for 1000 entries
	TrainLINE();
	return 0;
}
