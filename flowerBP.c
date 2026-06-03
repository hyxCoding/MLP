#include "flowerBP.h"
#include <stdio.h>
#include <string.h>
#define MAXN 1000
#define MAXLEN 20
#define IN1 4 //第一层输入数
#define IN2 8 //第二层输入数
#define OUT1 8 //第一层输出数
#define OUT2 3 //第二层输出数

struct data{//一行数据
    struct Matrix *p;//输入向量
    struct Matrix *t;//目标输出
};
char* path = "D:\\鸢尾花分类\\data.txt";//data路径
char* iris_1 = "Iris-setosa";//鸢尾花种类
char* iris_2 = "Iris-versicolor";
char* iris_3 = "Iris-virginica";
char type[MAXLEN];
int arr_size = 0;//数据总数量
int train_size = 0;//训练集数量
double a = 0.01;//学习速度
struct data arr_data[MAXN];//数据数组
struct Matrix *W1 = NULL;//第一层权值
struct Matrix *W2 = NULL;//第二层权值

struct Matrix *W1_u = NULL;//第一层权值更新矩阵
struct Matrix *W2_u = NULL;//第二层权值更新矩阵

struct Matrix *B1 = NULL;//第一层偏置值
struct Matrix *B2 = NULL;//第二层偏置值

struct Matrix *B1_u = NULL;//第一层偏置值更新矩阵
struct Matrix *B2_u = NULL;//第二层偏置值更新矩阵

struct Matrix *N1 = NULL;//第一层净输入
struct Matrix *N2 = NULL;//第二层净输入

struct Matrix *A1 = NULL;//第一层输出
struct Matrix *A2 = NULL;//第二层输出

struct Matrix *F1 = NULL;//第一层雅可比矩阵
struct Matrix *F2 = NULL;//第二层雅可比矩阵

struct Matrix *S1 = NULL;//第一层敏感度矩阵
struct Matrix *S2 = NULL;//第二层敏感度矩阵

struct Matrix *P1_T = NULL;//第一层输入转置
struct Matrix *A1_T = NULL;//第一层输出转置
struct Matrix *W2_T = NULL;//第二层权值矩阵转置

struct Matrix *Temp1 = NULL;//存储临时数据矩阵,8X3

void swap(struct data *a,struct data *b){//交换
    struct data temp = *a;
    *a = *b;
    *b = temp;
    return;
}

void assignT(struct data* d){//分配目标输出
    struct Matrix *t = newConstantMatrix(3,1,0);
    if(strcmp(iris_1,type) == 0){//是第一类
        t->arr[0] = 1;
    }else if(strcmp(iris_2,type) == 0){//是第二类
        t->arr[1] = 1;
    }else{//是第三类
        t->arr[2] = 1;
    }
    d->t = t;
    return;
}

void readData(){//读取数据
    FILE* file = fopen(path,"r");
    if(file == NULL){
        return;
    }
    while(1){
        struct Matrix *p = newMatrix(4,1);//输入向量
        if(fscanf(file,"%lf,%lf,%lf,%lf,%s\n",&(p->arr[0]),&(p->arr[1]),&(p->arr[2]),&(p->arr[3]),type) == 5){
            arr_data[arr_size].p = p;
            assignT(&arr_data[arr_size++]);
        }else{
            freeMatrix(&p);
            break;
        }
    }
    fclose(file);
    return;
}

void randomData(){//数据随机化分布
    int index;
    for(int i = 0;i < arr_size;i++){//随机化
        index = i + rand() % (arr_size - i);
        swap(&arr_data[i],&arr_data[index]);
    }
    return;
}

void forward_pass(int index){//正向传播
    matrixMultiplication(W1,arr_data[index].p,N1);//计算第一层净输入
    relu_matrix(N1,A1);//计算第一层输出
    matrixMultiplication(W2,A1,N2);//计算第二层净输入
    softmax(N2,A2);//计算第二层输出
    return;
}

void backward_pass(int index){//反向传播
    matrix_transpose(A1,A1_T);//计算A1转置
    matrix_transpose(W2,W2_T);//计算W2转置
    matrix_transpose(arr_data[index].p,P1_T);//计算p转置
    matrixSubtraction(A2,arr_data[index].t,S2);//计算S2
    scalarMultiplication(S2,B2_u,1.0);//计算B2_u
    matrixMultiplication(S2,A1_T,W2_u);//计算W2_u
    calculateJacobianMatrix(F1,N1,&relu_prime);//计算雅可比矩阵
    matrixMultiplication(F1,W2_T,Temp1);//计算S1
    matrixMultiplication(Temp1,S2,S1);//计算S1
    matrixMultiplication(S1,P1_T,W1_u);//计算W1_u
    scalarMultiplication(S1,B1_u,1.0);//计算B1_u
    scalarMultiplication(W1_u,W1_u,-a);//沿梯度反方向
    scalarMultiplication(B1_u,B1_u,-a);
    scalarMultiplication(W2_u,W2_u,-a);
    scalarMultiplication(B2_u,B2_u,-a);
    matrixAddition(W1,W1_u,W1);//更新参数
    matrixAddition(B1,B1_u,B1);
    matrixAddition(W2,W2_u,W2);
    matrixAddition(B2,B2_u,B2);
    return;
}

int maxIndex(struct Matrix *A){//最大值下标
    int size = A->row * A->col;
    int index = 0;
    double max_val = A->arr[0];
    for(int i = 0;i < size;i++){
        if(A->arr[i] > max_val){
            max_val = A->arr[i];
            index = i;
        }
    }
    return index;
}

int check(){//判断网络是否收敛
    int index1,index2;
    int ans = 0;
    for(int i = 0;i < train_size;i++){
        forward_pass(i);
        index1 = maxIndex(A2);
        index2 = maxIndex(arr_data[i].t);
        if(index1 == index2){//正确分类
            ans++;
        }
    }
    return ans;
}

int evaluate(){//检验网络性能
    int index1,index2;
    int ans = 0;
    for(int i = train_size;i < arr_size;i++){
        forward_pass(i);
        index1 = maxIndex(A2);
        index2 = maxIndex(arr_data[i].t);
        if(index1 == index2){//正确分类
            ans++;
        }
    }
    return ans;
}

int main(){
    srand(time(0));
    readData();
    randomData();
    W1_u = newMatrix(OUT1,IN1);
    B1_u = newMatrix(OUT1,1);
    W2_u = newMatrix(OUT2,IN2);
    B2_u = newMatrix(OUT2,1);

    N1 = newMatrix(OUT1,1);
    N2 = newMatrix(OUT2,1);
    A1 = newMatrix(OUT1,1);
    A2 = newMatrix(OUT2,1);
    S1 = newMatrix(OUT1,1);
    S2 = newMatrix(OUT2,1);
    F1 = newConstantMatrix(OUT1,OUT1,0);
    F2 = newConstantMatrix(OUT2,OUT2,0);

    A1_T = newMatrix(1,OUT1);
    W2_T = newMatrix(IN2,OUT2);
    P1_T = newMatrix(1,IN1);
    Temp1 = newMatrix(IN2,OUT2);
    train_size = arr_size * 2 / 3;//取三分之二训练
    while(1){
        W1 = newMatrixWithRandom(OUT1,IN1);//8输出,4输入
        B1 = newMatrixWithRandom(OUT1,1);
        W2 = newMatrixWithRandom(OUT2,IN2);//3输出,8输入
        B2 = newMatrixWithRandom(OUT2,1);
        for(int i = 0;i < 10000;i++){
            for(int i = 0;i < train_size;i++){
                forward_pass(i);
                backward_pass(i);
            }
        }
        int res = check();
        double ratio = (double)res / (double)train_size * 100.0;
        if(ratio > 95){//准确率高于95%结束训练
            printf("网络在训练集中正确分类了%d个数据,准确率为%.1f%%\n",res,ratio);
            break;
        }
        freeMatrix(&W1);//释放内存,选取新的随机值重新训练网络
        freeMatrix(&B1);
        freeMatrix(&W2);
        freeMatrix(&B2);
    }
    int res = evaluate();
    double ratio = (double)res / (double)(arr_size - train_size) * 100.0;
    printf("网络在测试集中正确分类了%d个数据,准确率为%.1f%%\n",res,ratio);
    return 0;
}
