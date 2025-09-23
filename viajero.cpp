#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <thread>
#include <iostream>
#include <mutex>
using namespace std;

// --- Parámetros editables ---
int NUM_NODES; // Cambia aquí el número de nodos
const int WINDOW_WIDTH = 900;
const int WINDOW_HEIGHT = 800;
const float NODE_RADIUS = 12.0f;

// --- Estructuras ---
struct Nodo {
    int id;
    float x, y;
};
vector<Nodo> nodos;
vector<vector<double>> distancias;

// --- Generación aleatoria de nodos ---
void generarNodos(int n) {
    random_device rd; mt19937 gen(rd());
    uniform_real_distribution<> disX(NODE_RADIUS, WINDOW_WIDTH-NODE_RADIUS);
    uniform_real_distribution<> disY(NODE_RADIUS, WINDOW_HEIGHT-NODE_RADIUS);

    nodos.clear();
    for (int i = 0; i < n; ++i) {
        nodos.push_back({i, (float)disX(gen), (float)disY(gen)});
    }
}

// --- Cálculo de matriz de distancias ---
void calcularDistancias() {
    int N = nodos.size();
    distancias.assign(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float dx = nodos[i].x - nodos[j].x;
            float dy = nodos[i].y - nodos[j].y;
            distancias[i][j] = sqrt(dx*dx + dy*dy);
        }
    }
}

// --- Calcula la aptitud de una ruta ---
double calcularAptitud(const vector<int>& ruta) {
    double suma = 0;
    int N = ruta.size();
    for (int i = 0; i < N-1; ++i) suma += distancias[ruta[i]][ruta[i+1]];
    suma += distancias[ruta[N-1]][ruta[0]]; // Regresa al inicio
    return 1.0 / suma;
}

// --- Dibuja círculo (nodo) ---
void dibujarNodo(float x, float y, float r, float cr, float cg, float cb) {
    glColor3f(cr, cg, cb);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x, y);
    for (int i = 0; i <= 30; ++i) {
        float t = 2.0f * 3.1415926f * i / 30;
        glVertex2f(x + cos(t)*r, y + sin(t)*r);
    }
    glEnd();
}

// --- Dibuja aristas (todas) ---
void dibujarAristas() {
    glColor3f(0.7f, 0.7f, 0.7f);
    glLineWidth(1.0);
    glBegin(GL_LINES);
    for (auto& n1 : nodos)
        for (auto& n2 : nodos)
            if (n1.id < n2.id)
                { glVertex2f(n1.x, n1.y); glVertex2f(n2.x, n2.y); }
    glEnd();
}

// --- Dibuja ruta ---
void dibujarRuta(const vector<int>& ruta) {
    glColor3f(0.2f, 0.8f, 1.0f);
    glLineWidth(3.0);
    glBegin(GL_LINE_STRIP);
    for (int idx : ruta) {
        glVertex2f(nodos[idx].x, nodos[idx].y);
    }
    glVertex2f(nodos[ruta[0]].x, nodos[ruta[0]].y); // Cierra ciclo
    glEnd();
}

// --- Renderizado completo ---
void render(const vector<int>& rutaActual = {}) {
    glClear(GL_COLOR_BUFFER_BIT);
    dibujarAristas();
    if (!rutaActual.empty()) dibujarRuta(rutaActual);
    for (auto& n : nodos) dibujarNodo(n.x, n.y, NODE_RADIUS, 1.0f, 0.3f, 0.3f);
}

// --- Ejemplo: calcula una ruta aleatoria y su aptitud ---
vector<int> mejorRuta;
double mejorAptitud = 0.0;

void buscarMejorRuta(int intentos = 2000) {
    int N = nodos.size();
    mejorAptitud = 0.0;
    mejorRuta.clear();
    vector<int> ruta(N);
    iota(ruta.begin(), ruta.end(), 0);

    mutex mtx;
    auto probar = [&](int inicio, int fin){
        vector<int> localRuta = ruta;
        double localMejor = 0.0; vector<int> localMejorRuta;
        random_device rd; mt19937 gen(rd());
        for (int i=inicio; i<fin; ++i) {
            shuffle(localRuta.begin(), localRuta.end(), gen);
            double apt = calcularAptitud(localRuta);
            if (apt > localMejor) { localMejor = apt; localMejorRuta = localRuta; }
        }
        if (localMejor > mejorAptitud) {
            lock_guard<mutex> lock(mtx);
            if (localMejor > mejorAptitud) {
                mejorAptitud = localMejor; mejorRuta = localMejorRuta;
            }
        }
    };
    int threads = min(8, intentos/200);
    vector<thread> ths;
    int chunk = intentos/threads;
    for (int t=0; t<threads; ++t)
        ths.emplace_back(probar, t*chunk, (t+1)*chunk);
    for(auto& th:ths) th.join();
}

// --- Main ---
int main() {
    cout << "Problema del Agente Viajero\n";
    cout << "Cantidad de nodos: "; cin >> NUM_NODES;

    generarNodos(NUM_NODES);
    calcularDistancias();
    buscarMejorRuta(4000);

    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "TSP OpenGL", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1);
    glMatrixMode(GL_MODELVIEW);

    while (!glfwWindowShouldClose(window)) {
        render(mejorRuta);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();

    cout << "\nMejor ruta encontrada (aleatoria): ";
    for(int v:mejorRuta) cout << v << " ";
    cout << mejorRuta[0] << " (regresa al inicio)\n";
    cout << "Aptitud (mayor es mejor): " << mejorAptitud << endl;
    return 0;
}