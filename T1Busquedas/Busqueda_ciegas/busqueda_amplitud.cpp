#include <iostream>
#include <queue>
#include <vector>
#include <unordered_map>
#include <unordered_set>
//#include <random>
using namespace std;

void bfs(int inicio, int objetivo, unordered_map<int, vector<int>>& grafo){
    queue<int> cola;

    unordered_set<int> visitados;
    unordered_map<int,int> padre;

    cola.push(inicio);
    visitados.insert(inicio);
    padre[inicio] = -1;
    
    while(!cola.empty()){
        int nodo = cola.front();
        cola.pop();

        if(nodo == objetivo){
            cout << "Camino encontrado\n";
            vector<int> camino;
            while(nodo != -1){
                camino.push_back(nodo);
                nodo = padre[nodo];
            }
        }
    }


}

void crearGrafo(int inicio, int objetivo, unordered_map<int,vector<int>>& grafo){

}

int main(){
    int n; cin >> n;
    for (int i = 1; i < n*n + 1;i++){

    }

    return 0;
}