
#ifndef SHEWCHUK_H
#define SHEWCHUK_H

extern "C" {

  void exactinit();
  float orient2d(float *pa, float *pb, float *pc);
  float orient2dfast(float *pa, float *pb, float *pc);
  float orient3d(float *pa, float *pb, float *pc);
  float orient3dfast(float *pa, float *pb, float *pc);
  float incircle(float *pa, float *pb, float *pc, float *pd);
  float incirclefast(float *pa, float *pb, float *pc, float *pd);
  float insphere(float *pa, float *pb, float *pc, float *pd, float *pe);
  float inspherefast(float *pa, float *pb, float *pc, float *pd, float *pe);

}

#endif // SHEWCHUK_H
