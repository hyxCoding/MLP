#ifndef FLOWERBP_H
#define FLOWERBP_H
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct Matrix{//矩阵
    int row;
    int col;
    double *arr;
};

struct Matrix *newMatrix(int row,int col){//构造新矩阵
    int size = row * col;
    struct Matrix *matrix = (struct Matrix*)malloc(sizeof(struct Matrix));
    matrix->row = row;
    matrix->col = col;
    matrix->arr = (double*)malloc(sizeof(double) * size);
    return matrix;
}

struct Matrix *newMatrixWithRandom(int row,int col){//构造新矩阵,并初始化为较小随机值
    int size = row * col;
    struct Matrix *matrix = newMatrix(row,col);
    for(int i = 0;i < size;i++){//初始化为随机较小值
        matrix->arr[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
    }
    return matrix;
}

struct Matrix *newConstantMatrix(int row,int col,double val){//创建带初始值的矩阵
    int size = row * col;
    struct Matrix *matrix = newMatrix(row,col);
    for(int i = 0;i < size;i++){
        matrix->arr[i] = val;
    }    
    return matrix;
}

struct Matrix *copyMatrix(struct Matrix *A){//矩阵复制
    int size = A->row * A->col;
    struct Matrix *res_matrix = newMatrix(A->row,A->col);
    for(int i = 0;i < size;i++){
        res_matrix->arr[i] = A->arr[i];
    }
    return res_matrix;
}

void freeMatrix(struct Matrix **matrix){//释放matrix
    if(*matrix == NULL){
        return;
    }
    free((*matrix)->arr);
    free(*matrix);
    *matrix = NULL;
    return;
}

void matrixMultiplication(struct Matrix *A,struct Matrix *B,struct Matrix *C){//矩阵相乘,AXB,存储到C
    if(!A || !B || !C){
        return;
    }
    int row = A->row;
    int col = B->col;
    int cnt = A->col;
    double res_mul = 0;
    for(int i = 0;i < row;i++){
        for(int j = 0;j < col;j++){
            res_mul = 0;
            for(int k = 0;k < cnt;k++){
                res_mul += A->arr[i * cnt + k] * B->arr[k * col + j];
            }
            C->arr[i * col + j] = res_mul;
        }
    }
    return;
}

void matrixAddition(struct Matrix *A,struct Matrix *B,struct Matrix *C){//矩阵相加,A+B
    if(!A || !B || !C){//无法相加
        return;
    }
    int size = A->row * A->col;
    for(int i = 0;i < size;i++){
        C->arr[i] = A->arr[i] + B->arr[i];
    }
    return;
}

void matrixSubtraction(struct Matrix *A,struct Matrix *B,struct Matrix *C){//矩阵相减,A-B
    if(!A || !B || !C){//无法相减
        return;
    }
    int size = A->row * A->col;
    for(int i = 0;i < size;i++){
        C->arr[i] = A->arr[i] - B->arr[i];
    }
    return;
}

void scalarMultiplication(struct Matrix *A,struct Matrix *C,double val){//矩阵乘以常数
    if(!A || !C){
        return;
    }
    int size = A->row * A->col;
    for(int i = 0;i < size;i++){
        C->arr[i] = A->arr[i] * val;
    }
    return;
}

void calculateJacobianMatrix(struct Matrix *A,struct Matrix *N,double(*f)(double)){//计算雅可比矩阵
    if(!A || !N ){
        return;
    }
    int row = A->row;
    int col = A->col;
    for(int i = 0;i < row;i++){
        A->arr[i * col + i] = (*f)(N->arr[i]);
    }
    return;
}

void matrix_transpose(struct Matrix *A,struct Matrix *C){//矩阵转置
    int row = A->row;
    int col = A->col;
    int index_a,index_c;
    for(int i = 0;i < row;i++){
        for(int j = 0;j < col;j++){
            index_a = i * col + j;
            index_c = j * row + i;
            C->arr[index_c] = A->arr[index_a];
        }
    }
    return;
}


double hardlim(double x){
    if(x >= 0){
        return 1;
    }
    return 0;
}

double relu(double x){
    return x > 0 ? x : 0;
}

void relu_matrix(struct Matrix *A,struct Matrix *C){
    int size = A->row * A->col;
    for(int i = 0;i < size;i++){
        C->arr[i] = relu(A->arr[i]);
    }
    return;
}

double relu_prime(double x){//导函数
    if(x > 0){
        return 1;
    }
    return 0;
}

double cross_entropy(struct Matrix *A,struct Matrix *B){//交叉熵
    //A是one-hot向量,B是softmax输出向量
    double ans = 0;
    int size = A->row;//单列矩阵
    for(int i = 0;i < size;i++){
        ans += A->arr[i] * log(B->arr[i]);
    }
    return -ans;
}

void softmax(struct Matrix *A,struct Matrix *C){//非线性函数
    double sum = 0;
    double max_val = A->arr[0];
    int size = A->row;//A是单列矩阵
    for(int i = 0;i < size;i++){
        max_val = max_val > A->arr[i] ? max_val : A->arr[i];
    }
    for(int i = 0;i < size;i++){
        C->arr[i] = exp(A->arr[i] - max_val);
        sum += C->arr[i];
    }
    for(int i=0;i<size;i++){
        C->arr[i] /= sum;
    }
    return;
}


#endif