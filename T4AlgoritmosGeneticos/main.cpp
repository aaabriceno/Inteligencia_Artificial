#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <string>
#include <numeric>

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>

// --- Configuración ---
const int POPULATION_SIZE = 10;
const int NUM_GENERATIONS = 100;
const int CHROMOSOME_LENGTH = 14;
const int GENE_LENGTH = 7;
const double MUTATION_RATE = 0.010;
const int TOURNAMENT_SIZE = 5;

// --- Estructuras de datos ---
struct Individual {
    std::vector<bool> chromosome;
    double fitness;
};

// --- Variables globales ---
std::vector<Individual> population(POPULATION_SIZE);
std::vector<double> bestValueHistory;
std::vector<double> avgValueHistory;
std::mt19937 g_gen;
int bestX, bestY;
double bestValue;
double executionTime;

// --- Función objetivo ---
double objectiveFunction(double x, double y) {
    return (x * x) - (y * y) + (2 * x * y);
}

// --- Decodificación corregida ---
int binaryToInt(const std::vector<bool>& gene) {
    int value = 0;
    for (bool bit : gene) {
        value = (value << 1) | bit;
    }
    if (value > 63) {
        return value - 128;
    }
    return value;
}

void decodeChromosome(const Individual& ind, int& x, int& y) {
    std::vector<bool> xGene(ind.chromosome.begin(), ind.chromosome.begin() + GENE_LENGTH);
    std::vector<bool> yGene(ind.chromosome.begin() + GENE_LENGTH, ind.chromosome.end());
    x = binaryToInt(xGene);
    y = binaryToInt(yGene);
}

// --- Evaluación de fitness para minimización ---
double calculateFitness(const Individual& ind) {
    int x, y;
    decodeChromosome(ind, x, y);
    double result = objectiveFunction(x, y);
    return 10000.0 / (1.0 + std::abs(result));
}

// --- Inicialización ---
void initializePopulation() {
    std::uniform_int_distribution<> dis(0, 1);
    for (auto& ind : population) {
        ind.chromosome.resize(CHROMOSOME_LENGTH);
        for (int j = 0; j < CHROMOSOME_LENGTH; ++j) {
            ind.chromosome[j] = dis(g_gen);
        }
    }
}

// --- Selección por torneo ---
Individual tournamentSelection(int tournamentSize) {
    std::uniform_int_distribution<> dis(0, POPULATION_SIZE - 1);
    Individual best = population[dis(g_gen)];
    for (int i = 1; i < tournamentSize; ++i) {
        Individual candidate = population[dis(g_gen)];
        if (candidate.fitness > best.fitness) {
            best = candidate;
        }
    }
    return best;
}

// --- Cruzamiento optimizado ---
std::vector<Individual> singlePointCrossover(const Individual& parent1, const Individual& parent2) {
    std::uniform_int_distribution<> dis(1, CHROMOSOME_LENGTH - 1);
    int crossoverPoint = dis(g_gen);

    Individual child1, child2;
    child1.chromosome.resize(CHROMOSOME_LENGTH);
    child2.chromosome.resize(CHROMOSOME_LENGTH);

    std::copy(parent1.chromosome.begin(), parent1.chromosome.begin() + crossoverPoint, child1.chromosome.begin());
    std::copy(parent2.chromosome.begin() + crossoverPoint, parent2.chromosome.end(), child1.chromosome.begin() + crossoverPoint);

    std::copy(parent2.chromosome.begin(), parent2.chromosome.begin() + crossoverPoint, child2.chromosome.begin());
    std::copy(parent1.chromosome.begin() + crossoverPoint, parent1.chromosome.end(), child2.chromosome.begin() + crossoverPoint);

    return {child1, child2};
}

// --- Mutación ---
void mutate(Individual& ind, double mutationRate) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < ind.chromosome.size(); ++i) {
        if (dis(g_gen) < mutationRate) {
            ind.chromosome[i] = !ind.chromosome[i];
        }
    }
}

// --- Algoritmo genético principal ---
void runGeneticAlgorithm() {
    std::random_device rd;
    g_gen.seed(rd());

    initializePopulation();
    bestValueHistory.clear();
    avgValueHistory.clear();

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        // Evaluación paralela
        std::vector<std::thread> threads;
        int numThreads = std::max(1, (int)std::thread::hardware_concurrency());
        int chunkSize = POPULATION_SIZE / numThreads;

        for (int i = 0; i < numThreads; ++i) {
            int start = i * chunkSize;
            int end = (i == numThreads - 1) ? POPULATION_SIZE : start + chunkSize;
            threads.emplace_back([start, end]() {
                for (int j = start; j < end; ++j) {
                    population[j].fitness = calculateFitness(population[j]);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }

        // Estadísticas
        double bestFitness = -1.0;
        double totalValue = 0.0;
        Individual currentBestInd;

        for (const auto& ind : population) {
            if (ind.fitness > bestFitness) {
                bestFitness = ind.fitness;
                currentBestInd = ind;
            }
            int x, y;
            decodeChromosome(ind, x, y);
            totalValue += objectiveFunction(x, y);
        }
        
        decodeChromosome(currentBestInd, bestX, bestY);
        bestValueHistory.push_back(objectiveFunction(bestX, bestY));
        avgValueHistory.push_back(totalValue / POPULATION_SIZE);

        // Nueva población con elitismo
        std::vector<Individual> newPopulation;
        newPopulation.reserve(POPULATION_SIZE);
        
        auto populationSorted = population;
        std::sort(populationSorted.begin(), populationSorted.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness > b.fitness;
            });
        
        for (int i = 0; i < 3 && i < POPULATION_SIZE; ++i) {
            newPopulation.push_back(populationSorted[i]);
        }

        while (newPopulation.size() < POPULATION_SIZE) {
            Individual parent1 = tournamentSelection(TOURNAMENT_SIZE);
            Individual parent2 = tournamentSelection(TOURNAMENT_SIZE);
            std::vector<Individual> children = singlePointCrossover(parent1, parent2);
            
            for (auto& child : children) {
                mutate(child, MUTATION_RATE);
                if (newPopulation.size() < POPULATION_SIZE) {
                    newPopulation.push_back(child);
                }
            }
        }
        population = newPopulation;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    executionTime = std::chrono::duration<double>(endTime - startTime).count();

    auto bestIt = std::max_element(population.begin(), population.end(),
        [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });
    decodeChromosome(*bestIt, bestX, bestY);
    bestValue = objectiveFunction(bestX, bestY);
}

// --- Funciones de dibujo OpenGL ---
void drawLine(float x1, float y1, float x2, float y2, float r, float g, float b, float thickness = 1.0f) {
    glColor3f(r, g, b);
    glLineWidth(thickness);
    glBegin(GL_LINES);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
    glLineWidth(1.0f);
}

void drawMinimizationGraph(GLFWwindow* window, int width, int height) {
    if (bestValueHistory.empty() || avgValueHistory.empty()) return;

    float graphPadding = 80.0f;
    float graphWidth = width - 2 * graphPadding;
    float graphHeight = height - 2 * graphPadding;
    
    // Encontrar los valores mínimo y máximo en ambas listas
    double minValue = *std::min_element(bestValueHistory.begin(), bestValueHistory.end());
    minValue = std::min(minValue, *std::min_element(avgValueHistory.begin(), avgValueHistory.end()));
    
    double maxValue = *std::max_element(bestValueHistory.begin(), bestValueHistory.end());
    maxValue = std::max(maxValue, *std::max_element(avgValueHistory.begin(), avgValueHistory.end()));
    
    double valueRange = maxValue - minValue;
    if (valueRange < 1e-10) valueRange = 1.0;

    // Dibujar ejes
    drawLine(graphPadding, height - graphPadding, width - graphPadding, height - graphPadding, 1.0f, 1.0f, 1.0f, 2.0f);
    drawLine(graphPadding, graphPadding, graphPadding, height - graphPadding, 1.0f, 1.0f, 1.0f, 2.0f);

    // Dibuja línea para el valor óptimo (0)
    if (minValue <= 0 && maxValue >= 0) {
        float y_optimo = graphPadding + (graphHeight - ((0 - minValue) * graphHeight / valueRange));
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(1, 0x00FF);
        drawLine(graphPadding, y_optimo, width - graphPadding, y_optimo, 1.0f, 0.5f, 0.0f);
        glDisable(GL_LINE_STIPPLE);
    }
    
    // Dibujar mejor valor (verde)
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINE_STRIP);
    for (size_t i = 0; i < bestValueHistory.size(); ++i) {
        float x = graphPadding + (i * graphWidth / (bestValueHistory.size() - 1));
        float y = graphPadding + (graphHeight - ((bestValueHistory[i] - minValue) * graphHeight / valueRange));
        glVertex2f(x, y);
    }
    glEnd();

    // Dibujar valor promedio (azul)
    glColor3f(0.0f, 0.5f, 1.0f);
    glBegin(GL_LINE_STRIP);
    for (size_t i = 0; i < avgValueHistory.size(); ++i) {
        float x = graphPadding + (i * graphWidth / (avgValueHistory.size() - 1));
        float y = graphPadding + (graphHeight - ((avgValueHistory[i] - minValue) * graphHeight / valueRange));
        glVertex2f(x, y);
    }
    glEnd();
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// --- Función principal ---
int main() {
    std::cout << "==============================================================\n";
    std::cout << " ALGORITMO GENETICO - MINIMIZACION f(x,y) = x² - y² + 2xy\n";
    std::cout << "==============================================================\n";
    std::cout << "Poblacion: " << POPULATION_SIZE << " individuos\n";
    std::cout << "Generaciones: " << NUM_GENERATIONS << "\n";
    std::cout << "==============================================================\n";
    
    runGeneticAlgorithm();

    std::cout << "\n=== RESULTADO FINAL ===\n";
    std::cout << "Mejor solucion: x = " << bestX << ", y = " << bestY << "\n";
    std::cout << "Valor minimo: f(x,y) = " << bestValue << "\n";
    std::cout << "Tiempo ejecucion: " << executionTime << " segundos\n";
    std::cout << "==============================================================\n";

    if (!glfwInit()) {
        std::cerr << "Error al inicializar GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(1200, 800, "Evolucion - Minimizacion f(x,y) = x² - y² + 2xy", NULL, NULL);
    if (!window) {
        std::cerr << "Error al crear ventana GLFW\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width, height, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        drawMinimizationGraph(window, width, height);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}