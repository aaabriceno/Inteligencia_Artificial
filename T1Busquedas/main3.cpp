#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <stack>
#include <thread>
#include <mutex>
#include <atomic>
#include <limits> // Para std::numeric_limits

// --- Alias para coordenadas lógicas y de renderizado ---
using Point = std::pair<int, int>;
using PointF = std::pair<float, float>; // Para coordenadas de punto flotante en OpenGL

// Estructura para el algoritmo A*
struct Node {
    Point position;
    double cost;
    double heuristic;
    bool operator>(const Node& other) const {
        return (cost + heuristic) > (other.cost + other.heuristic);
    }
};

// --- Estructura de Estado con Grilla Lógica ---
struct GraphState {
    int windowWidth = 1200;
    int windowHeight =1200;

    // --- Grilla con buen espaciado para que sea visible ---
    int gridWidth = 100; // 80x80 = 6,400 nodos. Cada celda tendrá 10x10 píxeles.
    int gridHeight = 100;

    std::vector<Point> vertices;
    std::map<Point, std::vector<Point>> adjacencyList;
    std::vector<Point> path;
    Point startNode, goalNode;
    bool nodesSelected = false;
};

// --- Sincronización de Hilos ---
std::mutex g_state_mutex;
std::atomic<bool> g_should_terminate_thread(false);

// Prototipos
void generateGraph(GraphState& state);
void removeNodes(GraphState& state, int percentage);
std::vector<Point> searchDFS(GraphState& state, Point start, Point goal);
std::vector<Point> searchBFS(GraphState& state, Point start, Point goal);
std::vector<Point> searchHillClimbing(GraphState& state, Point start, Point goal);
std::vector<Point> searchAStar(GraphState& state, Point start, Point goal);
void showMenu(GraphState& state);

double heuristic(Point a, Point b) {
    return std::sqrt(std::pow(a.first - b.first, 2) + std::pow(a.second - b.second, 2));
}

void generateGraph(GraphState& state) {
    state.vertices.clear();
    state.adjacencyList.clear();
    state.vertices.reserve(state.gridWidth * state.gridHeight);

    for (int x = 0; x < state.gridWidth; ++x) {
        for (int y = 0; y < state.gridHeight; ++y) {
            state.vertices.emplace_back(x, y);
        }
    }

    // --- ¡CORREGIDO! Conexiones en 8 direcciones (incluyendo diagonales) ---
    for (int x = 0; x < state.gridWidth; ++x) {
        for (int y = 0; y < state.gridHeight; ++y) {
            Point current = {x, y};
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue; // No es vecino de sí mismo

                    int nx = x + dx;
                    int ny = y + dy;

                    // Verificar si el vecino está dentro de los límites de la grilla
                    if (nx >= 0 && nx < state.gridWidth && ny >= 0 && ny < state.gridHeight) {
                        state.adjacencyList[current].push_back({nx, ny});
                    }
                }
            }
        }
    }
}

void removeNodes(GraphState& state, int percentage) {
    // (Sin cambios)
    if (percentage <= 0 || percentage >= 100) return;
    std::vector<Point> allVertices = state.vertices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(allVertices.begin(), allVertices.end(), gen);
    int nodesToRemoveCount = (allVertices.size() * percentage) / 100;
    std::set<Point> removedNodes(allVertices.begin(), allVertices.begin() + nodesToRemoveCount);
    state.vertices.erase(std::remove_if(state.vertices.begin(), state.vertices.end(),
        [&](const Point& v) { return removedNodes.count(v); }), state.vertices.end());
    for (const auto& removedNode : removedNodes) {
        state.adjacencyList.erase(removedNode);
    }
    for (auto& pair : state.adjacencyList) {
        pair.second.erase(std::remove_if(pair.second.begin(), pair.second.end(),
            [&](const Point& neighbor) { return removedNodes.count(neighbor); }), pair.second.end());
    }
}

// --- Algoritmos de Búsqueda (sin cambios) ---
std::vector<Point> searchDFS(GraphState& state, Point start, Point goal) {
    std::stack<Point> stack;
    std::map<Point, Point> cameFrom;
    std::set<Point> visited;
    stack.push(start);
    visited.insert(start);
    while (!stack.empty()) {
        Point current = stack.top();
        stack.pop();
        if (current == goal) break;
        if (state.adjacencyList.count(current)) {
            for (const auto& neighbor : state.adjacencyList.at(current)) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    cameFrom[neighbor] = current;
                    stack.push(neighbor);
                }
            }
        }
    }
    std::vector<Point> resultPath;
    if (cameFrom.find(goal) == cameFrom.end()) return {};
    for (Point at = goal; at != start; at = cameFrom[at]) { resultPath.push_back(at); }
    resultPath.push_back(start);
    std::reverse(resultPath.begin(), resultPath.end());
    return resultPath;
}
std::vector<Point> searchBFS(GraphState& state, Point start, Point goal) {
    std::queue<Point> queue;
    std::map<Point, Point> cameFrom;
    std::set<Point> visited;
    queue.push(start);
    visited.insert(start);
    while (!queue.empty()) {
        Point current = queue.front();
        queue.pop();
        if (current == goal) break;
        if (state.adjacencyList.count(current)) {
            for (const auto& neighbor : state.adjacencyList.at(current)) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    cameFrom[neighbor] = current;
                    queue.push(neighbor);
                }
            }
        }
    }
    std::vector<Point> resultPath;
    if (cameFrom.find(goal) == cameFrom.end()) return {};
    for (Point at = goal; at != start; at = cameFrom[at]) { resultPath.push_back(at); }
    resultPath.push_back(start);
    std::reverse(resultPath.begin(), resultPath.end());
    return resultPath;
}
std::vector<Point> searchHillClimbing(GraphState& state, Point start, Point goal) {
    std::vector<Point> resultPath;
    std::set<Point> visited;
    Point current = start;
    resultPath.push_back(current);
    visited.insert(current);
    while (current != goal) {
        Point next = current;
        double bestHeuristic = heuristic(current, goal);
        if (state.adjacencyList.count(current)) {
            for (const auto& neighbor : state.adjacencyList.at(current)) {
                if (visited.find(neighbor) != visited.end()) continue;
                double h = heuristic(neighbor, goal);
                if (h < bestHeuristic) {
                    bestHeuristic = h;
                    next = neighbor;
                }
            }
        }
        if (next == current) break;
        visited.insert(next);
        resultPath.push_back(next);
        current = next;
    }
    if (current != goal) return {};
    return resultPath;
}
std::vector<Point> searchAStar(GraphState& state, Point start, Point goal) {
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> openSet;
    std::map<Point, double> gScore;
    std::map<Point, Point> cameFrom;
    gScore[start] = 0;
    openSet.push({ start, 0, heuristic(start, goal) });
    while (!openSet.empty()) {
        Point current = openSet.top().position;
        openSet.pop();
        if (current == goal) break;
        if (state.adjacencyList.count(current)) {
            for (const auto& neighbor : state.adjacencyList.at(current)) {
                double tentative_gScore = gScore[current] + heuristic(current, neighbor);
                if (gScore.find(neighbor) == gScore.end() || tentative_gScore < gScore[neighbor]) {
                    gScore[neighbor] = tentative_gScore;
                    cameFrom[neighbor] = current;
                    openSet.push({ neighbor, tentative_gScore, heuristic(neighbor, goal) });
                }
            }
        }
    }
    std::vector<Point> resultPath;
    if (cameFrom.find(goal) == cameFrom.end()) return {};
    for (Point at = goal; at != start; at = cameFrom[at]) { resultPath.push_back(at); }
    resultPath.push_back(start);
    std::reverse(resultPath.begin(), resultPath.end());
    return resultPath;
}

// --- FUNCIÓN DE RENDERIZADO (con los colores correctos) ---
void render(const GraphState& state) {
    glClear(GL_COLOR_BUFFER_BIT);
    float cellWidth = (float)state.windowWidth / state.gridWidth;
    float cellHeight = (float)state.windowHeight / state.gridHeight;
    auto gridToGL = [&](Point p) -> PointF {
        float pixelX = p.first * cellWidth + cellWidth / 2.0f;
        float pixelY = p.second * cellHeight + cellHeight / 2.0f;
        float glX = (pixelX / state.windowWidth) * 2.0f - 1.0f;
        float glY = (pixelY / state.windowHeight) * 2.0f - 1.0f;
        return {glX, glY};
    };
    // 1. DIBUJAR LA GRILLA DE FONDO SIEMPRE EN BLANCO
    glColor3f(1.0f, 1.0f, 1.0f); // Color BLANCO
    glLineWidth(1.0);
    glBegin(GL_LINES);
    for (const auto& pair : state.adjacencyList) {
        PointF startGL = gridToGL(pair.first);
        for (const auto& end : pair.second) {
            // Optimización para no dibujar cada línea dos veces
            if (pair.first < end) {
                PointF endGL = gridToGL(end);
                glVertex2f(startGL.first, startGL.second);
                glVertex2f(endGL.first, endGL.second);
            }
        }
    }
    glEnd();
    // 2. DIBUJAR EL CAMINO ENCONTRADO EN AZUL
    if (!state.path.empty()) {
        glColor3f(0.0f, 0.5f, 1.0f);
        glLineWidth(3.0);
        glBegin(GL_LINE_STRIP);
        for (const auto& p : state.path) {
            PointF pGL = gridToGL(p);
            glVertex2f(pGL.first, pGL.second);
        }
        glEnd();
    }
    // 3. DIBUJAR NODOS DE INICIO Y FIN
    if (state.nodesSelected) {
        glPointSize(10.0);
        glBegin(GL_POINTS);
        PointF startGL = gridToGL(state.startNode);
        glColor3f(0.0f, 1.0f, 0.0f); // VERDE
        glVertex2f(startGL.first, startGL.second);
        PointF goalGL = gridToGL(state.goalNode);
        glColor3f(1.0f, 0.0, 0.0f); // ROJO
        glVertex2f(goalGL.first, goalGL.second);
        glEnd();
    }
}

int main() {
    if (!glfwInit()) return -1;
    GraphState state;
    GLFWwindow* window = glfwCreateWindow(state.windowWidth, state.windowHeight, "Graph Search - Malla Visible", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    generateGraph(state);
    std::cout << "Total de nodos creados: " << state.vertices.size() << std::endl;
    std::thread menuThread(showMenu, std::ref(state));
    while (!glfwWindowShouldClose(window) && !g_should_terminate_thread) {
        {
            std::lock_guard<std::mutex> lock(g_state_mutex);
            render(state);
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    g_should_terminate_thread = true;
    std::cout << "\nCerrando. Por favor, presione Enter en la consola para finalizar el programa." << std::endl;
    menuThread.join();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// --- FUNCIÓN DE MENÚ (sin cambios) ---
void showMenu(GraphState& state) {
    while (!g_should_terminate_thread) {
        int choice;
        std::cout << "\n=========== MENU ===========\n";
        std::cout << "1. Eliminar un porcentaje de nodos\n";
        std::cout << "2. Realizar una busqueda de camino\n";
        std::cout << "3. Salir del programa\n";
        std::cout << "============================\n";
        std::cout << "Seleccione una opcion: ";
        std::cin >> choice;
        if (std::cin.fail()) {
            std::cout << "Error: Entrada invalida. Por favor, ingrese solo un numero.\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (choice == 1) {
            int percentage;
            std::cout << "Ingrese el porcentaje de nodos a eliminar (1-99): ";
            std::cin >> percentage;
            if (std::cin.fail() || percentage <= 0 || percentage >= 100) {
                 std::cout << "Error: Porcentaje invalido.\n";
                 std::cin.clear();
                 std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                 continue;
            }
            std::lock_guard<std::mutex> lock(g_state_mutex);
            removeNodes(state, percentage);
            std::cout << "Nodos eliminados correctamente.\n";
        } else if (choice == 2) {
            int searchType;
            std::cout << "\n--- Seleccione Algoritmo de Busqueda ---\n";
            std::cout << "1. Profundidad (DFS)\n";
            std::cout << "2. Amplitud (BFS)\n";
            std::cout << "3. Escalada (Hill Climbing)\n";
            std::cout << "4. A* (A-Star)\n";
            std::cout << "Seleccione una opcion: ";
            std::cin >> searchType;
            if (std::cin.fail() || searchType < 1 || searchType > 4) {
                 std::cout << "Error: Opcion de algoritmo invalida.\n";
                 std::cin.clear();
                 std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                 continue;
            }
            int x1, y1, x2, y2;
            std::cout << "\nLa grilla es de " << state.gridWidth << "x" << state.gridHeight << " nodos.\n";
            std::cout << "Ingrese nodo inicial (x y) [ej: 0 0]: ";
            std::cin >> x1 >> y1;
            std::cout << "Ingrese nodo final (x y) [ej: " << state.gridWidth - 1 << " " << state.gridHeight - 1 << "]: ";
            std::cin >> x2 >> y2;
            if (std::cin.fail()) {
                 std::cout << "Error: Coordenadas invalidas.\n";
                 std::cin.clear();
                 std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                 continue;
            }
            if (x1 < 0 || x1 >= state.gridWidth || y1 < 0 || y1 >= state.gridHeight || x2 < 0 || x2 >= state.gridWidth || y2 < 0 || y2 >= state.gridHeight) {
                std::cout << "Error: Las coordenadas estan fuera de los limites de la grilla.\n";
                continue;
            }
            std::vector<Point> foundPath;
            Point start = {x1, y1}; Point goal = {x2, y2};
            if (searchType == 1) foundPath = searchDFS(state, start, goal);
            else if (searchType == 2) foundPath = searchBFS(state, start, goal);
            else if (searchType == 3) foundPath = searchHillClimbing(state, start, goal);
            else if (searchType == 4) foundPath = searchAStar(state, start, goal);
            std::lock_guard<std::mutex> lock(g_state_mutex);
            state.path = foundPath;
            state.startNode = start;
            state.goalNode = goal;
            state.nodesSelected = true;
            if (state.path.empty()) { std::cout << "Resultado: No se encontro un camino.\n"; }
            else { std::cout << "Resultado: Camino encontrado con " << state.path.size() << " nodos.\n"; }
        } else if (choice == 3) {
            std::cout << "Saliendo...\n";
            g_should_terminate_thread = true;
            break;
        } else {
            std::cout << "Opcion no valida. Intente de nuevo.\n";
        }
    }
}
