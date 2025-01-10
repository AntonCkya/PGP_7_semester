#include <iostream>
#include <cmath>
#include <iomanip>

int main(){
    int n;
    std::cin >> n;

    float * arr = (float *)malloc(n * sizeof(float));
    float x;

    for(int i = 0; i < n; i++){
        std::cin >> x;
        arr[i] = x;
    }

    bool flag = true;
    float temp;
    for(;;){
        flag = true;
        for(int i = 0; i < n - 1; i++){
            if(arr[i] > arr[i + 1]){
                temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                flag = false;
            }
        }
        if(flag){
            break;
        }
    }

    std::cout << std::scientific << std::setprecision(6);
    for(int i = 0; i < n; i++){
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}