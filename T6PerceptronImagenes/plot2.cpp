#include <GL/glut.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace std;

struct Point {
    double epoch;
    double train_loss;
    double val_loss;
};

vector<Point> lossData;  // <-- nombre cambiado
double max_loss = 0, min_loss = 1e9;
int window_width = 800, window_height = 600;

// Función para leer el CSV
void loadData(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error al abrir " << filename << endl;
        exit(1);
    }

    string line;
    getline(file, line); // saltar cabecera

    while (getline(file, line)) {
        stringstream ss(line);
        string epoch_s, train_s, val_s;
        getline(ss, epoch_s, ',');
        getline(ss, train_s, ',');
        getline(ss, val_s, ',');

        Point p;
        p.epoch = stod(epoch_s);
        p.train_loss = stod(train_s);
        p.val_loss = val_s.empty() ? 0.0 : stod(val_s);
        lossData.push_back(p);

        max_loss = max({max_loss, p.train_loss, p.val_loss});
        min_loss = min({min_loss, p.train_loss, p.val_loss});
    }
    file.close();
}

void drawText(float x, float y, const string& text) {
    glRasterPos2f(x, y);
    for (char c : text)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
}

void drawAxes() {
    glColor3f(1, 1, 1);
    glBegin(GL_LINES);
    // eje X
    glVertex2f(50, 50);
    glVertex2f(window_width - 50, 50);
    // eje Y
    glVertex2f(50, 50);
    glVertex2f(50, window_height - 50);
    glEnd();

    // Ticks y etiquetas
    glColor3f(0.7, 0.7, 0.7);
    int num_ticks_x = 10;
    int num_ticks_y = 10;

    double max_epoch = lossData.back().epoch;
    double step_x = (window_width - 100) / (double)num_ticks_x;
    double step_y = (window_height - 100) / (double)num_ticks_y;
    double step_val = (max_loss - min_loss) / num_ticks_y;

    for (int i = 0; i <= num_ticks_x; i++) {
        float x = 50 + i * step_x;
        glBegin(GL_LINES);
        glVertex2f(x, 45);
        glVertex2f(x, 55);
        glEnd();
        string label = to_string((int)(i * max_epoch / num_ticks_x));
        drawText(x - 10, 30, label);
    }

    for (int i = 0; i <= num_ticks_y; i++) {
        float y = 50 + i * step_y;
        glBegin(GL_LINES);
        glVertex2f(45, y);
        glVertex2f(55, y);
        glEnd();
        string label = to_string((float)(min_loss + i * step_val)).substr(0,5);
        drawText(10, y - 5, label);
    }

    drawText(window_width / 2 - 20, 15, "Epochs");
    drawText(10, window_height - 30, "Loss");
}

void drawCurve(const vector<Point>& lossData, bool isTrain) {
    if (lossData.empty()) return;

    glBegin(GL_LINE_STRIP);
    if (isTrain)
        glColor3f(0.2, 0.8, 0.2); // verde
    else
        glColor3f(0.9, 0.2, 0.2); // rojo

    double max_epoch = lossData.back().epoch;
    for (auto& p : lossData) {
        double x = 50 + (p.epoch / max_epoch) * (window_width - 100);
        double y = 50 + ((isTrain ? p.train_loss : p.val_loss) - min_loss) / (max_loss - min_loss) * (window_height - 100);
        glVertex2f(x, y);
    }
    glEnd();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    drawAxes();
    drawCurve(lossData, true);   // Train loss
    drawCurve(lossData, false);  // Validation loss

    // Leyenda
    glColor3f(1, 1, 1);
    drawText(window_width - 180, window_height - 70, "Leyenda:");
    glColor3f(0.2, 0.8, 0.2);
    drawText(window_width - 180, window_height - 85, "Train Loss");
    glColor3f(0.9, 0.2, 0.2);
    drawText(window_width - 180, window_height - 100, "Val Loss");

    glutSwapBuffers();
}

void reshape(int w, int h) {
    window_width = w;
    window_height = h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h);
    glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv) {
    loadData("loss_history.txt");

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Training and Validation Loss");

    glClearColor(0.1, 0.1, 0.1, 1.0);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0, window_width, 0, window_height);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMainLoop();
    return 0;
}
