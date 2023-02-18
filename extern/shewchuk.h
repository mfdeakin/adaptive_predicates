
#ifndef SHEWCHUK_H
#define SHEWCHUK_H

extern "C" {

    double orient2d(double *pa, double *pb, double *pc);
    double orient2dfast(double *pa, double *pb, double *pc);
    double orient3d(double *pa, double *pb, double *pc);
    double orient3dfast(double *pa, double *pb, double *pc);
    double incircle(double *pa, double *pb, double *pc, double *pd);
    double incirclefast(double *pa, double *pb, double *pc, double *pd);
    double insphere(double *pa, double *pb, double *pc, double *pd, double *pe);
    double inspherefast(double *pa, double *pb, double *pc, double *pd, double *pe);

}

#endif // SHEWCHUK_H
