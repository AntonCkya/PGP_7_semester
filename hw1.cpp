#include <iostream>
#include <cmath>
#include <iomanip>

int main(){
    float a, b, c;
    std::cin >> a >> b >> c;
    std::cout << std::fixed << std::setprecision(6);
    if(a == 0){
        if(b == 0){
            if(c == 0){
                std::cout << "any" << std::endl;
            } else {
                std::cout << "incorrect" << std::endl;
            }
        } else {
            std::cout << -c / b << std::endl;
        }
    } else {
        float D = b*b - 4*a*c;
        if(D < 0){
            std::cout << "imaginary" << std::endl;
        } else if (D == 0){
            std::cout << -b/(2 * a) << std::endl;
        } else {
            std::cout << (-b + std::sqrt(D))/(2 * a) << " " << (-b - std::sqrt(D))/(2 * a) << std::endl;
        }
    }
}