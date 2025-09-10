#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define NODE_RADIUS 20.0f
#define MAX_COLORS 3

float COLORS[MAX_COLORS][3] = {
    {1.0f, 0.0f, 0.0f}, //rojo
    {0.0f, 1.0f, 0.0f}, //verde
    {0.0f, 0.0f, 1.0f}  //azul
};

float GRAY[3] = { 0.5f, 0.5f, 0.5f };

int heuristic = 1; //1 = mas restrictiva, 2 = mas restringida
int backtrackCount = 0;
bool heuristicaActiva = false;

struct Nodo {
    int id;
    float x, y;
    int colorIndex; //-1 = sin color
    std::vector<int> vecinos;
};

struct Grafo {
    std::vector<Nodo> nodos;
    std::map<int, std::vector<int>> adyacencia;

    void agregarNodo(float x, float y) {
        Nodo nodo;
        nodo.id = nodos.size();
        nodo.x = x;
        nodo.y = y;
        nodo.colorIndex = -1; //color gris al inicio
        nodos.push_back(nodo);
    }

    bool esColorValido(int idNodo, int color) {
        for (int vecino : nodos[idNodo].vecinos) {
            if (nodos[vecino].colorIndex == color) {
                return false;
            }
        }
        return true;
    }

    std::vector<int> coloresDisponibles(int idNodo) {
        std::vector<int> disponibles;
        for (int c = 0; c < MAX_COLORS; ++c) {
            if (esColorValido(idNodo, c)) {
                disponibles.push_back(c);
            }
        }
        return disponibles;
    }

    void conectarNodos(int id1, int id2) {
        if (std::find(nodos[id1].vecinos.begin(), nodos[id1].vecinos.end(), id2) != nodos[id1].vecinos.end())
            return;

        nodos[id1].vecinos.push_back(id2);
        nodos[id2].vecinos.push_back(id1);
        adyacencia[id1].push_back(id2);
        adyacencia[id2].push_back(id1);
    }

    bool colorearNodo(int idNodo) {
        for (int c = 0; c < MAX_COLORS; ++c) {
            if (esColorValido(idNodo, c)) {
                nodos[idNodo].colorIndex = c;
                return true;
            }
        }
        return false;
    }

    bool colorearDesde(int index) {
        if (index >= nodos.size()) return true;
        if (nodos[index].vecinos.empty()) return colorearDesde(index + 1);

        std::vector<int> orden;
        for (int i = 0; i < nodos.size(); ++i) {
            if (nodos[i].vecinos.empty()) continue;
            orden.push_back(i);
        }

        if (heuristic == 1) {
            std::sort(orden.begin(), orden.end(), [&](int a, int b) {
                return coloresDisponibles(a).size() < coloresDisponibles(b).size();
                });
        }
        else {
            std::sort(orden.begin(), orden.end(), [&](int a, int b) {
                return nodos[a].vecinos.size() > nodos[b].vecinos.size();
                });
        }

        for (int i : orden) {
            if (nodos[i].colorIndex != -1) continue;
            for (int c = 0; c < MAX_COLORS; ++c) {
                if (esColorValido(i, c)) {
                    nodos[i].colorIndex = c;
                    if (colorearDesde(index + 1)) return true;
                    nodos[i].colorIndex = -1;
                    backtrackCount++;
                }
            }
            return false;
        }
        return true;
    }
    //
    int nodoClickeado(float mx, float my) {
        for (auto& nodo : nodos) {
            float dx = nodo.x - mx;
            float dy = nodo.y - my;
            if (sqrt(dx * dx + dy * dy) <= NODE_RADIUS)
                return nodo.id;
        }
        return -1;
    }


    void reiniciarColores() {
        for (auto& nodo : nodos) {
            nodo.colorIndex = -1;
        }
    }
};

Grafo grafo;
int nodoSeleccionado = -1;

void dibujarCirculo(float cx, float cy, float r, float color[3]) {
    glColor3fv(color);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(cx, cy);
    for (int i = 0; i <= 100; i++) {
        float angle = 2.0f * 3.1415926f * i / 100;
        glVertex2f(cx + cos(angle) * r, cy + sin(angle) * r);
    }
    glEnd();
}

void dibujarLinea(float x1, float y1, float x2, float y2) {
    glColor3f(0.2f, 0.2f, 0.2f);
    glLineWidth(1.5f);
    glBegin(GL_LINES);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        mx = (float)mx;
        my = WINDOW_HEIGHT - (float)my;

        int id = grafo.nodoClickeado(mx, my);
        if (id == -1) {
            grafo.agregarNodo(mx, my);
        }
        else {
            if (nodoSeleccionado == -1) {
                nodoSeleccionado = id;
            }
            else {
                if (id != nodoSeleccionado) {
                    grafo.conectarNodos(nodoSeleccionado, id);
                }
                nodoSeleccionado = -1;
            }
        }
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_1) {
            heuristic = 1;
            heuristicaActiva = true;
            backtrackCount = 0;
            grafo.reiniciarColores();
            grafo.colorearDesde(0);
            std::cout << "\nHeuristica usada: Variable mas restrictiva.\n";
            std::cout << "Backtracking realizado: " << backtrackCount << " veces\n";
        }
        else if (key == GLFW_KEY_2) {
            heuristic = 2;
            heuristicaActiva = true;
            backtrackCount = 0;
            grafo.reiniciarColores();
            grafo.colorearDesde(0);
            std::cout << "\nHeuristica usada: Variable mas restringida.\n";
            std::cout << "Backtracking realizado: " << backtrackCount << " veces\n";
        }
    }
}

void render() {
    glClear(GL_COLOR_BUFFER_BIT);

    for (auto& nodo : grafo.nodos) {
        for (int vecino : nodo.vecinos) {
            if (nodo.id < vecino) { //
                dibujarLinea(nodo.x, nodo.y, grafo.nodos[vecino].x, grafo.nodos[vecino].y);
            }
        }
    }

    for (auto& nodo : grafo.nodos) {
        if (nodo.colorIndex == -1) {
            dibujarCirculo(nodo.x, nodo.y, NODE_RADIUS, GRAY);
        }
        else {
            float* color = COLORS[nodo.colorIndex];
            dibujarCirculo(nodo.x, nodo.y, NODE_RADIUS, color);
        }
    }
}

int main() {
    std::srand(static_cast<unsigned>(std::time(0)));

    std::cout << "\nPresione 1 para la heuristica de la Variable mas restrictiva";
    std::cout << "\nPresione 2 para la heuristica de la Variable mas restringida\n";

    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Coloreo de Grafos", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetKeyCallback(window, keyCallback);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1);
    glMatrixMode(GL_MODELVIEW);

    while (!glfwWindowShouldClose(window)) {
        render();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}