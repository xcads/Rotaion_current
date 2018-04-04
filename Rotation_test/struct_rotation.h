#pragma once
#ifndef STRUCT_ROTATION_H
#define STRUCT_ROTATION_H

#define num_threads 1000000
#define block_width 1000
#define array_size 7

#define FILENAME        Coefficient.txt
#define PI				3.14159265358979323846264338327950288419716939937510       

#define BLOCK_LINES_NUMBER 1000

struct Objfunc {
	// xd = c1x + c2y
	double c1, c2;
};
using namespace std;
struct Point
{
	double x, y;
};

struct Line {
	double a1, a2, b;
	double slope;

	bool pruneFlag;
	int index;
};

struct AnswerXY
{
	int index1;
	struct Line line1;
	int index2;
	struct Line line2;

	struct Point interv;

	struct Line objfunc;
};
//typedef thrust::device_vector<int> intD;


typedef int BOOL;
typedef struct Point Point;
typedef struct Line Line;
typedef struct AnswerXY AnswerXY;
typedef struct Objfunc Objfunc;
struct AnswerXY answer;



#endif