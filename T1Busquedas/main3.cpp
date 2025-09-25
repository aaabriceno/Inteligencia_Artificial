#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glut.h>
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
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;
const float NODE_RADIUS = 12.0f;

bool debeSalir = false;
mutex mtx;

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
    uniform_int_distribution<> disX(0,100);
    uniform_int_distribution<> disY(0,100);

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

//Dibuja el numero de nodos sobre el circulo
void dibujarNumeroNodo(float x, float y, int numero){
    glColor3f(0,0,0); // Color negro para el texto
    glRasterPos2f(x-5, y-5); // x e y ya están escalados
    string numStr = to_string(numero);
    for (char c: numStr){
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
    }
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
            if (n1.id < n2.id) {
                float x1 = n1.x * (WINDOW_WIDTH / 100.0f);
                float y1 = n1.y * (WINDOW_HEIGHT / 100.0f);
                float x2 = n2.x * (WINDOW_WIDTH / 100.0f);
                float y2 = n2.y * (WINDOW_HEIGHT / 100.0f);
                glVertex2f(x1, y1); glVertex2f(x2, y2);
            }
    glEnd();
}

// --- Dibujar un plano cartesiano
void dibujarPlanoCartesiano() {
    glColor3f(0.2f, 0.2f, 0.2f);
    glLineWidth(2.0);
    glBegin(GL_LINES);
    // Eje X
    glVertex2f(0, WINDOW_HEIGHT/2);
    glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT/2);
    // Eje Y
    glVertex2f(WINDOW_WIDTH/2, 0);
    glVertex2f(WINDOW_WIDTH/2, WINDOW_HEIGHT);
    glEnd();
}

// --- Dibuja ruta ---
void dibujarRuta(const vector<int>& ruta) {
    glColor3f(0.2f, 0.8f, 1.0f);
    glLineWidth(3.0);
    glBegin(GL_LINE_STRIP);
    for (int idx : ruta) {
        float x = nodos[idx].x * (WINDOW_WIDTH / 100.0f);
        float y = nodos[idx].y * (WINDOW_HEIGHT / 100.0f);
        glVertex2f(x, y);
    }
    // Cierra ciclo
    float x0 = nodos[ruta[0]].x * (WINDOW_WIDTH / 100.0f);
    float y0 = nodos[ruta[0]].y * (WINDOW_HEIGHT / 100.0f);
    glVertex2f(x0, y0);
    glEnd();
}

// --- Renderizado completo ---
void render(const vector<int>& rutaActual = {}) {
    glClear(GL_COLOR_BUFFER_BIT);
    //dibujarPlanoCartesiano(); // Dibuja los ejes
    dibujarAristas();
    if (!rutaActual.empty()) dibujarRuta(rutaActual);
    for (auto& n : nodos){
    dibujarNodo(n.x * (WINDOW_WIDTH / 100.0f), n.y * (WINDOW_HEIGHT / 100.0f), NODE_RADIUS, 1.0f, 0.3f, 0.3f);
    dibujarNumeroNodo(n.x * (WINDOW_WIDTH / 100.0f), n.y * (WINDOW_HEIGHT / 100.0f), n.id);
    }
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

void limpiarGrafo(){
    nodos.clear();
    distancias.clear();
    mejorRuta.clear();
    mejorAptitud = 0.0;
}

void menuInteractivo() {
    while (!debeSalir) {
        int opcion;
        cout << "\n--- Menú ---\n";
        cout << "1. Generar nuevo grafo\n";
        cout << "2. Limpiar gráfica\n";
        cout << "3. Salir\n";
        cout << "Seleccione una opción: ";
        cin >> opcion;
        if (opcion == 1) {
            cout << "Cantidad de nodos: ";
            int n; cin >> n;
            std::lock_guard<std::mutex> lock(mtx);
            NUM_NODES = n;
            generarNodos(NUM_NODES);
            calcularDistancias();
            buscarMejorRuta(4000);
            cout << "Valores de x e y de los nodos generados:\n";
            for (const auto& n : nodos) {
                cout << "Nodo " << n.id << ": (" << n.x << ", " << n.y << ")\n";
            }
            cout << "\nMejor ruta encontrada (aleatoria): ";
            for(int v:mejorRuta) cout << v << " ";
            cout << mejorRuta[0] << " (regresa al inicio)\n";
            cout << "Aptitud (mayor es mejor): " << mejorAptitud << endl;
        } else if (opcion == 2) {
            std::lock_guard<std::mutex> lock(mtx);
            limpiarGrafo();
            cout << "Gráfica limpiada.\n";
        } else if (opcion == 3) {
            debeSalir = true;
            cout << "Saliendo...\n";
        } else {
            cout << "Opción inválida.\n";
        }
    }
}

int main() {
    int argc = 1; char* argv[1] = {(char*)"app"};
    glutInit(&argc, argv);

    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "TSP OpenGL", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1);
    glMatrixMode(GL_MODELVIEW);

    // Lanza el menú en un hilo
    thread menuThread(menuInteractivo);

    while (!glfwWindowShouldClose(window) && !debeSalir) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            if(!nodos.empty()){
                render(mejorRuta);
            }
            else{
                glClear(GL_COLOR_BUFFER_BIT);
            }
            
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
        this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    debeSalir = true;
    menuThread.join();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
