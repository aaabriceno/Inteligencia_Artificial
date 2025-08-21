#include <iostream>
#include <queue>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
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

            for (int i = camino.size() -1 ; i >=0; --i){
                cout << camino[i] << " ";
            }
            cout << endl;
            return;
        }

        for (int vecino: grafo[nodo]){
            if(visitados.find(vecino) == visitados.end()){
                cola.push(vecino);
                visitados.insert(vecino);
                padre[vecino] =nodo;
            }
        }   
    }
    cout << "No se encontro un camino \n";
}

unordered_map<int,vector<int>> eliminar_nodos(unordered_map<int,vector<int>> grafo){
    unordered_map<int,vector<int>> nuevo_grafo = grafo;
    double porcentaje = 0.3;
    size_t cantidad_nodos = grafo.size();

    size_t nodos_a_eliminar = static_cast<size_t>(ceil(cantidad_nodos * porcentaje));

    vector<int> lista_de_nodos;
    for (const auto& par : grafo) {
        lista_de_nodos.push_back(par.first);
    }

    unsigned seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 g(seed);
    shuffle(lista_de_nodos.begin(), lista_de_nodos.end(), g);
    
    for (size_t i = 0; i < nodos_a_eliminar; ++i) {
        int nodo_a_borrar = lista_de_nodos[i];
        
        nuevo_grafo.erase(nodo_a_borrar);

        for (auto& par : nuevo_grafo) {
            auto& vecinos = par.second;
            auto nuevo_fin = remove(vecinos.begin(), vecinos.end(), nodo_a_borrar);
            vecinos.erase(nuevo_fin, vecinos.end());
        }
    }
    return nuevo_grafo;
}

void crearGrafo(int n, unordered_map<int,vector<int>>& grafo){
    for (int i = 1; i <= n*n; ++i){
        if ( i % n != 0){
            grafo[i].push_back(i+1);
            grafo[i+1].push_back(i);
        }

        if(i + n <= n*n){
            grafo[i].push_back(i+n);
            grafo[i+n].push_back(i);
        }
    }
}

void imprimirGrafo(const unordered_map<int, vector<int>>& grafo) {
    if (grafo.empty()) {
        cout << "El grafo está vacío." << endl;
        return;
    }

    for (const auto& par : grafo) {
        cout << "Nodo " << par.first << " -> Vecinos: ";
        if (par.second.empty()) {
            cout << "Ninguno";
        } else {
            for (int vecino : par.second) {
                cout << vecino << " ";
            }
        }
        cout << endl;
    }
}

int main() {
    int n; 
    cout << "Ingrese el lado de la malla: ";
    cin >> n;
    
    unordered_map<int, vector<int>> grafo;
    crearGrafo(n, grafo);
    
    cout << "\n--- Grafo Original ---" << endl;
    imprimirGrafo(grafo);
    
    grafo = eliminar_nodos(grafo);

    cout << "\n--- Grafo Después de la Eliminación ---" << endl;
    imprimirGrafo(grafo);

    int inicio, objetivo;
    cout << "\nNodo de inicio: "; 
    cin >> inicio;
    cout << "Nodo objetivo: "; 
    cin >> objetivo;

    bfs(inicio, objetivo, grafo);

    return 0;
}
