#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <queue>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <stack>
#include <thread>

const int windowWidth = 400;
const int windowHeight = 200;
const int gridSpacing = 10;
std::vector<std::pair<int, int>> vertices;
std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> edges;
std::vector<std::pair<int, int>> path;
std::pair<int, int> startNode, goalNode;
bool nodesSelected = false;


struct Node {
    std::pair<int, int> position;
    double cost;
    double heuristic;
    bool operator>(const Node& other) const {
        return (cost + heuristic) > (other.cost + other.heuristic);
    }
};

double heuristic(std::pair<int, int> a, std::pair<int, int> b) {
    return std::sqrt(std::pow(a.first - b.first, 2) + std::pow(a.second - b.second, 2));
}

void generateVertices() {
    vertices.clear();
    edges.clear();
    for (int x = 0; x <= windowWidth; x += gridSpacing) {
        for (int y = 0; y <= windowHeight; y += gridSpacing) {
            vertices.emplace_back(x, y);
        }
    }
    for (const auto& [x, y] : vertices) {
        if (x + gridSpacing <= windowWidth) edges.push_back({ {x, y}, {x + gridSpacing, y} });
        if (y + gridSpacing <= windowHeight) edges.push_back({ {x, y}, {x, y + gridSpacing} });
        if (x + gridSpacing <= windowWidth && y + gridSpacing <= windowHeight) edges.push_back({ {x, y}, {x + gridSpacing, y + gridSpacing} });
        if (x - gridSpacing >= 0 && y + gridSpacing <= windowHeight) edges.push_back({ {x, y}, {x - gridSpacing, y + gridSpacing} });
    }
}

void removeNodes(int percentage) {
    int nodesToRemove = (vertices.size() * percentage) / 100;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(vertices.begin(), vertices.end(), gen);
    vertices.resize(vertices.size() - nodesToRemove);
    edges.erase(std::remove_if(edges.begin(), edges.end(), [&](auto& edge) {
        return std::find(vertices.begin(), vertices.end(), edge.first) == vertices.end() ||
            std::find(vertices.begin(), vertices.end(), edge.second) == vertices.end();
        }), edges.end());
}

std::vector<std::pair<int, int>> searchDFS(std::pair<int, int> start, std::pair<int, int> goal) {
    std::stack<std::pair<int, int>> stack;
    std::map<std::pair<int, int>, std::pair<int, int>> cameFrom;
    std::set<std::pair<int, int>> visited;

    stack.push(start);
    visited.insert(start);

    while (!stack.empty()) {
        auto current = stack.top();
        stack.pop();

        if (current == goal) break;

        for (auto& edge : edges) {
            std::pair<int, int> neighbor;
            if (edge.first == current) neighbor = edge.second;
            else if (edge.second == current) neighbor = edge.first;
            else continue;

            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                cameFrom[neighbor] = current;
                stack.push(neighbor);
            }
        }
    }

    //reconstruccion del camino
    std::vector<std::pair<int, int>> path;
    if (cameFrom.find(goal) == cameFrom.end()) return {};//cuando no se encuentra el camino

    for (auto at = goal; at != start; at = cameFrom[at]) {
        path.push_back(at);
    }
    path.push_back(start);
    std::reverse(path.begin(), path.end());

    return path;
}

std::vector<std::pair<int, int>> searchBFS(std::pair<int, int> start, std::pair<int, int> goal) {
    std::queue<std::pair<int, int>> queue;
    std::map<std::pair<int, int>, std::pair<int, int>> cameFrom;
    std::set<std::pair<int, int>> visited; //evitamos repetir nodos

    queue.push(start);
    visited.insert(start);

    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop();

        if (current == goal) break;

        for (auto& edge : edges) {
            std::pair<int, int> neighbor;
            if (edge.first == current) neighbor = edge.second;
            else if (edge.second == current) neighbor = edge.first;
            else continue;

            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                cameFrom[neighbor] = current;
                queue.push(neighbor);
            }
        }
    }

    //reconstruye el camino
    std::vector<std::pair<int, int>> path;
    if (cameFrom.find(goal) == cameFrom.end()) return {}; //cuando no hay camino

    for (auto at = goal; at != start; at = cameFrom[at]) {
        path.push_back(at);
    }
    path.push_back(start);
    std::reverse(path.begin(), path.end());

    return path;
}


std::vector<std::pair<int, int>> searchHillClimbing(std::pair<int, int> start, std::pair<int, int> goal) {
    std::vector<std::pair<int, int>> path;
    std::set<std::pair<int, int>> visited; //para evitar ciclos
    std::pair<int, int> current = start;

    path.push_back(current);
    visited.insert(current);

    while (current != goal) {
        std::pair<int, int> next = current;
        double bestHeuristic = heuristic(current, goal);

        for (auto& edge : edges) {
            std::pair<int, int> neighbor;
            if (edge.first == current) neighbor = edge.second;
            else if (edge.second == current) neighbor = edge.first;
            else continue;
            //
            if (visited.find(neighbor) != visited.end()) continue;

            double h = heuristic(neighbor, goal);
            if (h < bestHeuristic) {  //busca el mejor vecino con heuristica menor
                bestHeuristic = h;
                next = neighbor;
            }
        }

        if (next == current) break; //si no hay mejor vecino, se detiene

        visited.insert(next);
        path.push_back(next);
        current = next;
    }

    //verifica si llego al nodo final
    if (current != goal) return {}; //no encontro camino

    return path;
}


std::vector<std::pair<int, int>> searchAStar(std::pair<int, int> start, std::pair<int, int> goal) {
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> openSet;
    std::map<std::pair<int, int>, double> gScore;
    std::map<std::pair<int, int>, std::pair<int, int>> cameFrom;
    std::set<std::pair<int, int>> closedSet; //para evitar revisitar nodos

    openSet.push({ start, 0, heuristic(start, goal) });
    gScore[start] = 0;

    while (!openSet.empty()) {
        auto current = openSet.top().position;
        openSet.pop();

        if (current == goal) break; //si se llega al objetivo, termina

        if (closedSet.find(current) != closedSet.end()) continue; //si ya lo visitamos, lo ignoramos
        closedSet.insert(current);

        for (auto& edge : edges) {
            std::pair<int, int> neighbor;
            if (edge.first == current) neighbor = edge.second;
            else if (edge.second == current) neighbor = edge.first;
            else continue;

            if (closedSet.find(neighbor) != closedSet.end()) continue; //no revisitar nodos cerrados

            double tentative_gScore = gScore[current] + heuristic(current, neighbor);

            if (gScore.find(neighbor) == gScore.end() || tentative_gScore < gScore[neighbor]) {
                gScore[neighbor] = tentative_gScore;
                cameFrom[neighbor] = current;
                double fScore = tentative_gScore + heuristic(neighbor, goal);
                openSet.push({ neighbor, tentative_gScore, fScore });
            }
        }
    }

    //reconstruccion del camino
    std::vector<std::pair<int, int>> path;
    if (cameFrom.find(goal) == cameFrom.end()) return {}; //no se encontro el camino

    for (auto at = goal; at != start; at = cameFrom[at]) path.push_back(at);
    path.push_back(start);
    std::reverse(path.begin(), path.end());

    return path;
}

void showMenu() {
    while (true) {
        int choice;
        std::cout << "Seleccione una opcion:\n";
        std::cout << "1. Eliminar porcentaje de nodos\n";
        std::cout << "2. Realizar busqueda\n";
        std::cin >> choice;

        if (choice == 1) {
            int percentage;
            std::cout << "Ingrese el porcentaje de nodos a eliminar: ";
            std::cin >> percentage;
            removeNodes(percentage);
        }
        else if (choice == 2) {
            int searchType;
            std::cout << "Seleccione algoritmo de busqueda:\n";
            std::cout << "1. Profundidad (DFS)\n";
            std::cout << "2. Amplitud (BFS)\n";
            std::cout << "3. Hill Climbing\n";
            std::cout << "4. A*\n";
            std::cin >> searchType;

            int x1, y1, x2, y2;
            std::cout << "Ingrese nodo inicial (x y): ";
            std::cin >> x1 >> y1;
            std::cout << "Ingrese nodo final (x y): ";
            std::cin >> x2 >> y2;

            startNode = { x1, y1 };
            goalNode = { x2, y2 };
            nodesSelected = true;

            if (searchType == 1) path = searchDFS({ x1, y1 }, { x2, y2 });
            else if (searchType == 2) path = searchBFS({ x1, y1 }, { x2, y2 });
            else if (searchType == 3) path = searchHillClimbing({ x1, y1 }, { x2, y2 });
            else if (searchType == 4) path = searchAStar({ x1, y1 }, { x2, y2 });
            /*
            std::cout << "camino encontrado con " << path.size() << " nodos:\n";
            for (const auto& p : path) {
                std::cout << "(" << p.first << ", " << p.second << ") ";
            }
            */
            std::cout << std::endl;
        }
    }
}

//
void renderPath() {
    if (path.empty()) return;  //si no hay un camino, no dibuja

    glColor3f(1.0f, 0.0f, 0.0f); //cambio de color
    glLineWidth(2.5);//
    glBegin(GL_LINE_STRIP);
    for (const auto& p : path) {
        float x = (p.first / (float)windowWidth) * 2 - 1;
        float y = (p.second / (float)windowHeight) * 2 - 1;
        glVertex2f(x, y);
    }
    glEnd();
    glLineWidth(1.0);
}

//
void render() {
    glClear(GL_COLOR_BUFFER_BIT);

    //loss nodos
    /*
    glColor3f(0.0f, 1.0f, 0.0f);
    glPointSize(3.0);
    glBegin(GL_POINTS);
    for (const auto& vertex : vertices) {
        glVertex2f(vertex.first / (float)windowWidth * 2 - 1, vertex.second / (float)windowHeight * 2 - 1);
    }
    glEnd();
    */
    //lasaristas
    glColor3f(0.5f, 0.5f, 0.5f);
    glBegin(GL_LINES);
    for (const auto& edge : edges) {
        glVertex2f(edge.first.first / (float)windowWidth * 2 - 1, edge.first.second / (float)windowHeight * 2 - 1);
        glVertex2f(edge.second.first / (float)windowWidth * 2 - 1, edge.second.second / (float)windowHeight * 2 - 1);
    }
    glEnd();

    //dibujar solo el camino recorrido
    renderPath();

    glFlush();


    if (nodesSelected) {
        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(6.0);
        glBegin(GL_POINTS);
        glVertex2f(startNode.first / (float)windowWidth * 2 - 1, startNode.second / (float)windowHeight * 2 - 1);
        glVertex2f(goalNode.first / (float)windowWidth * 2 - 1, goalNode.second / (float)windowHeight * 2 - 1);
        glEnd();
    }
}

int main() {
    if (!glfwInit()) return -1;
    generateVertices();

    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Graph Search", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    std::thread menuThread(showMenu);

    while (!glfwWindowShouldClose(window)) {
        render();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    menuThread.join();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}