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
#include <limits>
#include <unordered_set> // Necesario para el BFS optimizado

// --- Alias para coordenadas lógicas y de renderizado ---
using Point = std::pair<int, int>;
using PointF = std::pair<float, float>;

// --- NUEVO: Hash para std::pair, para usar Point en unordered_set ---
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // Técnica estándar para combinar dos hashes
        return h1 ^ (h2 << 1);
    }
};

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
    int windowWidth = 700;
    int windowHeight = 700;
    int gridWidth = 50;
    int gridHeight = 50;

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

double heuristic_sq(Point a, Point b) {
    double dx = static_cast<double>(a.first - b.first);
    double dy = static_cast<double>(a.second - b.second);
    return dx * dx + dy * dy;
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

    for (int x = 0; x < state.gridWidth; ++x) {
        for (int y = 0; y < state.gridHeight; ++y) {
            Point current = {x, y};
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < state.gridWidth && ny >= 0 && ny < state.gridHeight) {
                        state.adjacencyList[current].push_back({nx, ny});
                    }
                }
            }
        }
    }
}

void removeNodes(GraphState& state, int percentage) {
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

// --- Algoritmo DFS (sin cambios) ---
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

// --- CORREGIDO: BFS optimizado con unordered_set ---
std::vector<Point> searchBFS(GraphState& state, Point start, Point goal) {
    std::queue<Point> queue;
    std::map<Point, Point> cameFrom;
    std::unordered_set<Point, pair_hash> visited; // <--- CAMBIO AQUÍ
    
    queue.push(start);
    visited.insert(start);
    
    while (!queue.empty()) {
        Point current = queue.front();
        queue.pop();
        if (current == goal) break;
        if (state.adjacencyList.count(current)) {
            for (const auto& neighbor : state.adjacencyList.at(current)) {
                if (visited.find(neighbor) == visited.end()) { // Búsqueda O(1)
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

// --- CORREGIDO: Hill Climbing robusto con backtracking y movimientos laterales ---
std::vector<Point> searchHillClimbing(GraphState& state, Point start, Point goal) {
    // Verificar si el nodo inicial o final fueron eliminados
    if (state.adjacencyList.find(start) == state.adjacencyList.end() || 
        state.adjacencyList.find(goal) == state.adjacencyList.end()) {
        std::cout << "Hill Climbing: El nodo inicial o final no existe en el grafo." << std::endl;
        return {};
    }

    if (start == goal) {
        return {start};
    }

    std::vector<Point> path;
    path.push_back(start);

    std::unordered_set<Point, pair_hash> path_set;
    path_set.insert(start);
    
    // --- INICIO DE LA LÓGICA NUEVA ---
    // Se establece un límite máximo de retrocesos para evitar que el algoritmo se congele
    const int max_backtracks = state.gridWidth * state.gridHeight; // Límite generoso
    int backtrack_counter = 0;
    // --- FIN DE LA LÓGICA NUEVA ---

    const int max_sideways_moves = 5;
    const int max_path_length = state.gridWidth * state.gridHeight;
    int sideways_moves_count = 0;

    while (!path.empty() && path.back() != goal) {
        Point current = path.back();

        if (path.size() > max_path_length) {
            std::cout << "Hill Climbing: Se excedio la longitud maxima del camino." << std::endl;
            return {};
        }

        Point best_neighbor = {-1, -1};
        Point sideways_neighbor = {-1, -1};
        double min_heuristic = heuristic_sq(current, goal);

        if (state.adjacencyList.count(current)) {
            for (const auto& neighbor : state.adjacencyList.at(current)) {
                if (!path_set.count(neighbor)) {
                    double neighbor_heuristic = heuristic_sq(neighbor, goal);
                    if (neighbor_heuristic < min_heuristic) {
                        min_heuristic = neighbor_heuristic;
                        best_neighbor = neighbor;
                        sideways_neighbor = {-1, -1};
                    } else if (best_neighbor.first == -1 && neighbor_heuristic == min_heuristic && sideways_moves_count < max_sideways_moves) {
                         sideways_neighbor = neighbor;
                    }
                }
            }
        }
        
        if (best_neighbor.first != -1) {
            path.push_back(best_neighbor);
            path_set.insert(best_neighbor);
            sideways_moves_count = 0;
        } else if (sideways_neighbor.first != -1) {
            path.push_back(sideways_neighbor);
            path_set.insert(sideways_neighbor);
            sideways_moves_count++;
        } else {
            // Atascado: no hay vecinos mejores, se debe retroceder (backtrack)
            path_set.erase(path.back());
            path.pop_back();
            sideways_moves_count = 0;

            // --- INICIO DE LA LÓGICA NUEVA ---
            // Incrementar el contador de retroceso y verificar si se ha superado el límite
            backtrack_counter++;
            if (backtrack_counter > max_backtracks) {
                std::cout << "\n--------------------------------------------------------------------------\n";
                std::cout << "ALGORITMO DETENIDO: Hill Climbing ha excedido el limite de retrocesos.\n";
                std::cout << "MOTIVO: El algoritmo quedo atrapado en un 'maximo local' complejo.\n";
                std::cout << "        Para salir, tendria que retroceder una cantidad excesiva de pasos,\n";
                std::cout << "        lo que es muy ineficiente. Se ha detenido la busqueda.\n";
                std::cout << "--------------------------------------------------------------------------\n";
                return {}; // Se devuelve un camino vacío para indicar el fallo y detenerlo.
            }
            // --- FIN DE LA LÓGICA NUEVA ---
        }
    }

    if (path.empty() || path.back() != goal) {
        // Este mensaje ahora aparecerá si el algoritmo retrocede hasta el inicio
        std::cout << "Hill Climbing: No se encontro un camino (retrocedio hasta el punto de partida).\n";
        return {};
    }

    return path;
}

// --- Algoritmo A* (sin cambios) ---
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

// --- Funciones de renderizado, main y menú (sin cambios) ---
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
    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(1.0);
    glBegin(GL_LINES);
    for (const auto& pair : state.adjacencyList) {
        PointF startGL = gridToGL(pair.first);
        for (const auto& end : pair.second) {
            if (pair.first < end) {
                PointF endGL = gridToGL(end);
                glVertex2f(startGL.first, startGL.second);
                glVertex2f(endGL.first, endGL.second);
            }
        }
    }
    glEnd();
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
    if (state.nodesSelected) {
        glPointSize(10.0);
        glBegin(GL_POINTS);
        PointF startGL = gridToGL(state.startNode);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex2f(startGL.first, startGL.second);
        PointF goalGL = gridToGL(state.goalNode);
        glColor3f(1.0f, 0.0, 0.0f);
        glVertex2f(goalGL.first, goalGL.second);
        glEnd();
    }
}

int main() {
    if (!glfwInit()) return -1;
    GraphState state;
    GLFWwindow* window = glfwCreateWindow(state.windowWidth, state.windowHeight, "Graph Search - Algoritmos Mejorados", NULL, NULL);
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
            else if (searchType == 3) {
                // --- INICIO DE LA LÓGICA MODIFICADA ---
                std::cout << "\nIntentando con Hill Climbing...\n";
                foundPath = searchHillClimbing(state, start, goal);
            
                // --- FIN DE LA LÓGICA MODIFICADA ---
            }
            else if (searchType == 4) foundPath = searchAStar(state, start, goal);

            // Actualizar el estado para el renderizado
            std::lock_guard<std::mutex> lock(g_state_mutex);
            state.path = foundPath;
            state.startNode = start;
            state.goalNode = goal;
            state.nodesSelected = true;
            if (state.path.empty()) { std::cout << "Resultado: No se encontro un camino (incluso con A*).\n"; }
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
