#include <iostream>
#include <queue>
#include <stack>
#include <set>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
using namespace std;

/*
 * Implementación de distintos algoritmos de búsqueda sobre una malla
 * bidimensional de tamaño N×N. Cada celda se representa mediante un
 * par de enteros (fila, columna). El grafo es no ponderado y permite
 * desplazamientos en las ocho direcciones (verticales, horizontales y
 * diagonales). Se incluyen las siguientes búsquedas:
 *  1. Búsqueda en amplitud (BFS) – garantiza el camino de menor
 *     longitud en número de aristas【318959941191139†L106-L121】.
 *  2. Búsqueda en profundidad iterativa (DFS) – explora tan profundo
 *     como sea posible antes de retroceder【669078328113471†L98-L115】.
 *  3. Búsqueda Hill Climbing – búsqueda local guiada por una heurística
 *     (distancia Chebyshev) que puede quedar atrapada en óptimos
 *     locales.
 *  4. Búsqueda A* – búsqueda informada que utiliza el coste real y
 *     heurístico para encontrar el camino óptimo cuando la heurística
 *     es admisible y consistente【318959941191139†L106-L121】.
 */

// Hash personalizado para pair<int,int> que permite usarlo como clave en
// unordered_map y unordered_set. Combina los hashes de ambos elementos.
struct PairHash {
    size_t operator()(const pair<int,int>& p) const noexcept {
        return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
    }
};

// Heurística Chebyshev: devuelve la máxima diferencia absoluta entre
// coordenadas. Es admisible para desplazamientos en ocho direcciones
// cuando todas las aristas tienen coste unitario.
int heuristic(const pair<int,int>& a, const pair<int,int>& b) {
    return max(abs(a.first - b.first), abs(a.second - b.second));
}

// Búsqueda en amplitud (BFS). Encuentra el camino más corto en número de
// aristas entre inicio y objetivo en grafos no ponderados【318959941191139†L106-L121】.
void bfs(pair<int,int> inicio,
         pair<int,int> objetivo,
         unordered_map<pair<int,int>, vector<pair<int,int>>, PairHash>& grafo) {
    queue<pair<int,int>> cola;
    unordered_set<pair<int,int>, PairHash> visitados;
    unordered_map<pair<int,int>, pair<int,int>, PairHash> padre;

    cola.push(inicio);
    visitados.insert(inicio);
    padre[inicio] = {-1, -1};

    while (!cola.empty()) {
        pair<int,int> nodo = cola.front();
        cola.pop();
        if (nodo == objetivo) {
            cout << "Camino encontrado\n";
            vector<pair<int,int>> camino;
            // reconstruir el camino
            while (nodo != make_pair(-1,-1)) {
                camino.push_back(nodo);
                nodo = padre[nodo];
            }
            // imprimir en orden inverso
            for (int i = camino.size() - 1; i >= 0; --i) {
                cout << "(" << camino[i].first << ", " << camino[i].second << ") ";
            }
            cout << endl;
            return;
        }
        for (const auto& vecino : grafo[nodo]) {
            if (visitados.find(vecino) == visitados.end()) {
                cola.push(vecino);
                visitados.insert(vecino);
                padre[vecino] = nodo;
            }
        }
    }
    cout << "No se encontró un camino\n";
}

// Búsqueda en profundidad (DFS) iterativa. Se utiliza una pila explícita.
void dfs(pair<int,int> inicio,
                  pair<int,int> objetivo,
                  unordered_map<pair<int,int>, vector<pair<int,int>>, PairHash>& grafo) {
    stack<pair<int,int>> pila;
    unordered_set<pair<int,int>, PairHash> visitados;
    unordered_map<pair<int,int>, pair<int,int>, PairHash> padre;

    pila.push(inicio);
    visitados.insert(inicio);
    padre[inicio] = {-1, -1};

    while (!pila.empty()) {
        pair<int,int> nodo = pila.top();
        pila.pop();
        if (nodo == objetivo) {
            cout << "Camino encontrado\n";
            vector<pair<int,int>> camino;
            while (nodo != make_pair(-1,-1)) {
                camino.push_back(nodo);
                nodo = padre[nodo];
            }
            for (int i = camino.size() - 1; i >= 0; --i) {
                cout << "(" << camino[i].first << ", " << camino[i].second << ") ";
            }
            cout << endl;
            return;
        }
        for (const auto& vecino : grafo[nodo]) {
            if (visitados.find(vecino) == visitados.end()) {
                pila.push(vecino);
                visitados.insert(vecino);
                padre[vecino] = nodo;
            }
        }
    }
    cout << "No se encontró un camino\n";
}

// Búsqueda Hill Climbing. Selecciona siempre el vecino con mejor heurística
// (menor distancia al objetivo) y avanza hasta que no haya mejora.
void hillClimbing(pair<int,int> inicio,
                  pair<int,int> objetivo,
                  unordered_map<pair<int,int>, vector<pair<int,int>>, PairHash>& grafo) {
    unordered_set<pair<int,int>, PairHash> visitados;
    unordered_map<pair<int,int>, pair<int,int>, PairHash> padre;
    pair<int,int> actual = inicio;
    visitados.insert(actual);
    padre[actual] = {-1,-1};

    while (true) {
        // Si alcanzamos el objetivo, reconstruimos el camino
        if (actual == objetivo) {
            cout << "Camino encontrado\n";
            vector<pair<int,int>> camino;
            pair<int,int> nodo = actual;
            while (nodo != make_pair(-1,-1)) {
                camino.push_back(nodo);
                nodo = padre[nodo];
            }
            for (int i = camino.size()-1; i >= 0; --i) {
                cout << "(" << camino[i].first << ", " << camino[i].second << ") ";
            }
            cout << endl;
            return;
        }
        // Evaluar el heurístico del nodo actual
        int h_actual = heuristic(actual, objetivo);
        pair<int,int> mejor_vecino = actual;
        int mejor_h = h_actual;
        // Buscar el vecino con mejor heurística (estrictamente menor)
        for (const auto& vecino : grafo[actual]) {
            if (visitados.find(vecino) != visitados.end()) continue;
            int h_vecino = heuristic(vecino, objetivo);
            if (h_vecino < mejor_h) {
                mejor_h = h_vecino;
                mejor_vecino = vecino;
            }
        }
        // Si ningún vecino mejora la heurística, no se puede avanzar
        if (mejor_vecino == actual) {
            cout << "No se encontró un camino\n";
            return;
        }
        // Avanzar al mejor vecino
        padre[mejor_vecino] = actual;
        visitados.insert(mejor_vecino);
        actual = mejor_vecino;
    }
}

// Búsqueda A* (A Star). Utiliza g(n) + h(n) para ordenar la exploración.
void aStar(pair<int,int> inicio,
           pair<int,int> objetivo,
           unordered_map<pair<int,int>, vector<pair<int,int>>, PairHash>& grafo) {
    // Estructura para la prioridad (f, nodo)
    struct Nodo {
        int f;
        pair<int,int> node;
        bool operator>(const Nodo& other) const {
            return f > other.f;
        }
    };
    priority_queue<Nodo, vector<Nodo>, greater<Nodo>> abiertos;
    unordered_set<pair<int,int>, PairHash> cerrados;
    unordered_map<pair<int,int>, pair<int,int>, PairHash> padre;
    unordered_map<pair<int,int>, int, PairHash> gScore;
    gScore[inicio] = 0;
    abiertos.push({heuristic(inicio, objetivo), inicio});
    padre[inicio] = {-1,-1};
    while (!abiertos.empty()) {
        auto actual = abiertos.top().node;
        abiertos.pop();
        if (cerrados.find(actual) != cerrados.end()) continue;
        // Si encontramos el objetivo, reconstruir el camino
        if (actual == objetivo) {
            cout << "Camino encontrado\n";
            vector<pair<int,int>> camino;
            pair<int,int> nodo = actual;
            while (nodo != make_pair(-1,-1)) {
                camino.push_back(nodo);
                nodo = padre[nodo];
            }
            for (int i = camino.size() - 1; i >= 0; --i) {
                cout << "(" << camino[i].first << ", " << camino[i].second << ") ";
            }
            cout << endl;
            return;
        }
        cerrados.insert(actual);
        int g_actual = gScore[actual];
        for (const auto& vecino : grafo[actual]) {
            if (cerrados.find(vecino) != cerrados.end()) continue;
            int g_tentativo = g_actual + 1;
            // Si no hay registro de gScore del vecino o encontramos un camino mejor
            auto it = gScore.find(vecino);
            if (it == gScore.end() || g_tentativo < it->second) {
                gScore[vecino] = g_tentativo;
                padre[vecino] = actual;
                int f = g_tentativo + heuristic(vecino, objetivo);
                abiertos.push({f, vecino});
            }
        }
    }
    cout << "No se encontró un camino\n";
}

// Elimina un porcentaje de nodos del grafo de forma aleatoria.
unordered_map<pair<int,int>, vector<pair<int,int>>, PairHash>
eliminar_nodos(int porcentaje,unordered_map<pair<int,int>, vector<pair<int,int>>, PairHash>& grafo) {
    unordered_map<pair<int,int>, vector<pair<int,int>>, PairHash> nuevo = grafo;
    size_t total = grafo.size();
    size_t a_borrar = static_cast<size_t>(ceil(total * porcentaje / 100.0));
    vector<pair<int,int>> nodos;
    nodos.reserve(total);
    for (const auto& par : grafo) {
        nodos.push_back(par.first);
    }
    unsigned seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 rng(seed);
    shuffle(nodos.begin(), nodos.end(), rng);
    for (size_t i = 0; i < a_borrar && i < nodos.size(); ++i) {
        auto key = nodos[i];
        nuevo.erase(key);
        for (auto& par : nuevo) {
            auto& vec = par.second;
            auto new_end = remove(vec.begin(), vec.end(), key);
            vec.erase(new_end, vec.end());
        }
    }
    return nuevo;
}

// Crea una malla n×n donde cada nodo está conectado a sus ocho vecinos.
void crearGrafo(int n,unordered_map<pair<int,int>, vector<pair<int,int>>, PairHash>& grafo) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            pair<int,int> nodo = {i, j};
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    if (di == 0 && dj == 0) continue;
                    int ni = i + di;
                    int nj = j + dj;
                    if (ni >= 0 && ni < n && nj >= 0 && nj < n) {
                        grafo[nodo].push_back({ni, nj});
                    }
                }
            }
        }
    }
}

// Imprime el grafo con sus conexiones.
void imprimirGrafo(const unordered_map<pair<int,int>, vector<pair<int,int>>, PairHash>& grafo) {
    if (grafo.empty()) {
        cout << "El grafo está vacío." << endl;
        return;
    }
    for (const auto& par : grafo) {
        cout << "Nodo (" << par.first.first << ", " << par.first.second << ") -> Vecinos: ";
        if (par.second.empty()) {
            cout << "Ninguno";
        } else {
            for (const auto& vecino : par.second) {
                cout << "(" << vecino.first << ", " << vecino.second << ") ";
            }
        }
        cout << endl;
    }
}

int main() {
    // Leer tamaño de la malla
    int n = 50;
    // Crear el grafo completo
    unordered_map<pair<int,int>, vector<pair<int,int>>, PairHash> grafo;
    crearGrafo(n, grafo);
    cout << "\n--- Grafo Original ---" << endl;
    imprimirGrafo(grafo);
    // Eliminar un porcentaje de nodos
    int porcentaje;
    cout << "Ingrese el porcentaje de nodos a eliminar: ";
    cin >> porcentaje;
    grafo = eliminar_nodos(porcentaje, grafo);
    cout << "\n--- Grafo Después de la Eliminación ---" << endl;
    imprimirGrafo(grafo);
    // Interacción con el usuario para buscar caminos
    pair<int,int> inicio, objetivo;
    char continuar = 's';
    while (continuar != 'n' && continuar != 'N') {
        // Elegir método
        int opcion;
        cout << "\nSeleccione el método de búsqueda:\n";
        cout << " 1. Búsqueda en amplitud (BFS)\n";
        cout << " 2. Búsqueda en profundidad (DFS)\n";
        cout << " 3. Hill Climbing\n";
        cout << " 4. A*\n";
        cout << "Ingrese el número de la opción: ";
        cin >> opcion;
        // Leer nodos de inicio y objetivo
        cout << "Nodo de inicio (fila columna): ";
        cin >> inicio.first >> inicio.second;
        if (grafo.find(inicio) == grafo.end()) {
            cout << "El nodo de inicio no existe en el grafo.\n";
            continue;
        }
        cout << "Nodo objetivo (fila columna): ";
        cin >> objetivo.first >> objetivo.second;
        if (grafo.find(objetivo) == grafo.end()) {
            cout << "El nodo objetivo no existe en el grafo.\n";
            continue;
        }
        // Ejecutar búsqueda según opción
        switch (opcion) {
            case 1:
                bfs(inicio, objetivo, grafo);
                break;
            case 2:
                dfs(inicio, objetivo, grafo);
                break;
            case 3:
                hillClimbing(inicio, objetivo, grafo);
                break;
            case 4:
                aStar(inicio, objetivo, grafo);
                break;
            default:
                cout << "Opción no válida.\n";
                break;
        }
        cout << "\n¿Desea hacer otra consulta? (s/n): ";
        cin >> continuar;
    }
    return 0;
}
