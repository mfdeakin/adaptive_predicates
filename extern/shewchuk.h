
#ifndef SHEWCHUK_H
#define SHEWCHUK_H

extern "C" {

  void exactinit();
  double orient2d(const double *pa, const double *pb, const double *pc);
  double orient2dfast(const double *pa, const double *pb, const double *pc);
  double orient3d(const double *pa, const double *pb, const double *pc);
  double orient3dfast(const double *pa, const double *pb, const double *pc);
  double incircle(const double *pa, const double *pb, const double *pc, const double *pd);
  double incirclefast(const double *pa, const double *pb, const double *pc, const double *pd);
  double insphere(const double *pa, const double *pb, const double *pc, const double *pd, const double *pe);
  double inspherefast(const double *pa, const double *pb, const double *pc, const double *pd, const double *pe);

}

#endif // SHEWCHUK_H
