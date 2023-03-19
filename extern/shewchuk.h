
#ifndef SHEWCHUK_H
#define SHEWCHUK_H

extern "C" {

  void exactinit();
  float orient2d(const float *pa, const float *pb, const float *pc);
  float orient2dfast(const float *pa, const float *pb, const float *pc);
  float orient3d(const float *pa, const float *pb, const float *pc);
  float orient3dfast(const float *pa, const float *pb, const float *pc);
  float incircle(const float *pa, const float *pb, const float *pc, const float *pd);
  float incirclefast(const float *pa, const float *pb, const float *pc, const float *pd);
  float insphere(const float *pa, const float *pb, const float *pc, const float *pd, const float *pe);
  float inspherefast(const float *pa, const float *pb, const float *pc, const float *pd, const float *pe);

}

#endif // SHEWCHUK_H
