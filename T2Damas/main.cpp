#include <GL/glut.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <string>
using namespace std;

enum Equipo { ROJO, NEGRO };

struct Ficha {
    bool activa;
    Equipo equipo;
};

struct Movimiento {
    int f0, c0, f1, c1;
    bool captura;
};

const int N = 8;
Ficha tablero[N][N];
Equipo turno = ROJO;
int profundidadMax = 3; 

void inicializar() {
    for (int f = 0; f < N; f++) {
        for (int c = 0; c < N; c++) {
            tablero[f][c].activa = false;
        }
    }
    for (int f = 0; f < 3; f++) {
        for (int c = 0; c < N; c++) {
            if ((f + c) % 2 == 1) {
                tablero[f][c] = {true, ROJO};
            }
        }
    }
    for (int f = 5; f < 8; f++) {
        for (int c = 0; c < N; c++) {
            if ((f + c) % 2 == 1) {
                tablero[f][c] = {true, NEGRO};
            }
        }
    }
}

int evaluar(Equipo quien, Ficha est[N][N]) {
    int score = 0;
    for (int f = 0; f < N; f++) {
        for (int c = 0; c < N; c++) {
            if (est[f][c].activa) {
                if (est[f][c].equipo == quien) score++;
                else score--;
            }
        }
    }
    return score;
}

void copiar(Ficha a[N][N], Ficha b[N][N]) {
    for (int f = 0; f < N; f++)
        for (int c = 0; c < N; c++)
            b[f][c] = a[f][c];
}

void aplicar(Ficha est[N][N], Movimiento m) {
    est[m.f1][m.c1] = est[m.f0][m.c0];
    est[m.f0][m.c0].activa = false;
    if (m.captura) {
        int fm = (m.f0 + m.f1) / 2;
        int cm = (m.c0 + m.c1) / 2;
        est[fm][cm].activa = false;
    }
}

vector<Movimiento> generar(Equipo eq, Ficha est[N][N]) {
    vector<Movimiento> movs;
    int dir = (eq == ROJO) ? 1 : -1;
    for (int f = 0; f < N; f++) {
        for (int c = 0; c < N; c++) {
            if (est[f][c].activa && est[f][c].equipo == eq) {
                // mover simple
                for (int dc = -1; dc <= 1; dc += 2) {
                    int nf = f + dir, nc = c + dc;
                    if (nf >= 0 && nf < N && nc >= 0 && nc < N && !est[nf][nc].activa) {
                        movs.push_back({f, c, nf, nc, false});
                    }
                }
                // captura
                for (int dc = -2; dc <= 2; dc += 4) {
                    int nf = f + 2 * dir, nc = c + dc;
                    int mf = f + dir, mc = c + dc / 2;
                    if (nf >= 0 && nf < N && nc >= 0 && nc < N &&
                        !est[nf][nc].activa && est[mf][mc].activa &&
                        est[mf][mc].equipo != eq) {
                        movs.push_back({f, c, nf, nc, true});
                    }
                }
            }
        }
    }
    return movs;
}

int minimax(Ficha est[N][N], int prof, bool maxTurno, Equipo ia) {
    Equipo actual = maxTurno ? ia : (ia == ROJO ? NEGRO : ROJO);
    vector<Movimiento> movs = generar(actual, est);
    if (prof == 0 || movs.empty()) {
        return evaluar(ia, est);
    }
    if (maxTurno) {
        int mejor = -999;
        for (auto m : movs) {
            Ficha copia[N][N]; copiar(est, copia);
            aplicar(copia, m);
            mejor = max(mejor, minimax(copia, prof - 1, false, ia));
        }
        return mejor;
    } else {
        int peor = 999;
        for (auto m : movs) {
            Ficha copia[N][N]; copiar(est, copia);
            aplicar(copia, m);
            peor = min(peor, minimax(copia, prof - 1, true, ia));
        }
        return peor;
    }
}

Movimiento mejorMovimiento(Equipo ia) {
    vector<Movimiento> movs = generar(ia, tablero);
    int mejorVal = -999, idx = 0;
    for (int i = 0; i < (int)movs.size(); i++) {
        Ficha copia[N][N]; copiar(tablero, copia);
        aplicar(copia, movs[i]);
        int val = minimax(copia, profundidadMax, false, ia); // 🔹 usa variable global
        if (val > mejorVal) {
            mejorVal = val; idx = i;
        }
    }
    return movs[idx];
}

// ======== GRAFICOS ========

int casillaSelF = -1, casillaSelC = -1;

void dibujar() {
    glClear(GL_COLOR_BUFFER_BIT);

    // tablero
    for (int f = 0; f < N; f++) {
        for (int c = 0; c < N; c++) {
            if ((f + c) % 2 == 0) glColor3f(0.9, 0.9, 0.9);
            else glColor3f(0.3, 0.3, 0.3);
            glBegin(GL_QUADS);
            glVertex2f(c, f);
            glVertex2f(c + 1, f);
            glVertex2f(c + 1, f + 1);
            glVertex2f(c, f + 1);
            glEnd();

            if (tablero[f][c].activa) {
                if (tablero[f][c].equipo == ROJO) glColor3f(0.8, 0.1, 0.1);
                else glColor3f(0, 0, 0);
                float cx = c + 0.5, cy = f + 0.5;
                glBegin(GL_TRIANGLE_FAN);
                glVertex2f(cx, cy);
                for (int k = 0; k <= 30; k++) {
                    float ang = k * 2 * 3.14159 / 30;
                    glVertex2f(cx + 0.4 * std::cos(ang), cy + 0.4 * std::sin(ang));
                }
                glEnd();
            }
        }
    }

    glutSwapBuffers();
}

void click(int boton, int estado, int x, int y) {
    if (boton == GLUT_LEFT_BUTTON && estado == GLUT_DOWN && turno == ROJO) {
        int c = x / (600 / N);
        int f = (600 - y) / (600 / N);
        if (casillaSelF == -1) {
            if (tablero[f][c].activa && tablero[f][c].equipo == ROJO) {
                casillaSelF = f; casillaSelC = c;
            }
        } else {
            vector<Movimiento> movs = generar(ROJO, tablero);
            for (auto m : movs) {
                if (m.f0 == casillaSelF && m.c0 == casillaSelC && m.f1 == f && m.c1 == c) {
                    aplicar(tablero, m);
                    turno = NEGRO;
                    Movimiento ia = mejorMovimiento(NEGRO);
                    aplicar(tablero, ia);
                    turno = ROJO;
                    break;
                }
            }
            casillaSelF = casillaSelC = -1;
        }
    }
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    if (argc > 1) {
        string dif = argv[1];
        if (dif == "facil") profundidadMax = 1;
        else if (dif == "normal") profundidadMax = 3;
        else if (dif == "dificil") profundidadMax = 5;
        else {
            cout << "Uso: ./main <facil|normal|dificil>\n";
            return 1;
        }
    }

    cout << "Dificultad: " << profundidadMax << " niveles de profundidad\n";

    inicializar();
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(600, 600);
    glutCreateWindow("Damas simples");
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0, N, 0, N);
    glutDisplayFunc(dibujar);
    glutMouseFunc(click);
    glutMainLoop();
    return 0;
}