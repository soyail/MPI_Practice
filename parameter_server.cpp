#include <mpi.h>
#include <vector>
#include <iostream>
#include <cstdlib>

std::vector<float> generate_random_gradient(int size){
    std::vector<float> gradient(size);
    for(int i=0; i<size; i++) gradient[i] = static_cast<float>(rand())/RAND_MAX;
    return gradient;
}

void print_vector(std::vector<float>& vec){
    for(float val:vec){
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv); //初始MPI环境

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //获取当前进程id
    MPI_Comm_size(MPI_COMM_WORLD, &size); //获取总进程数
    std::cout << "rank: " << rank << std::endl;
    // 模型参数
    const int parameter_size = 10;
    std::vector<float> model_parameters(parameter_size, 0.0f);

    if(rank == 0){
        for(int i=1; i<size; i++){
            std::vector<float> worker_grad(parameter_size);
            MPI_Recv(worker_grad.data(), parameter_size, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout <<"[Parameter Server] Received gradient: ";
            print_vector(worker_grad);

            float learning_rate = 0.1f;
            // param = param - learning_rate * grad
            for(int j=0; j<parameter_size; j++){
                model_parameters[j] -= (learning_rate * worker_grad[j])/size;
            }
        }
        print_vector(model_parameters);
    }
    else{
        srand(time(nullptr) + rank);
        std::vector<float> local_gradient(parameter_size);
        local_gradient = generate_random_gradient(parameter_size);
        print_vector(local_gradient);
        MPI_Send(local_gradient.data(), parameter_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}