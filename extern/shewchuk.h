
#ifndef SHEWCHUK_H
#define SHEWCHUK_H

namespace Shewchuk {

extern "C" {

void exactinit();
/* Predicate implementations:
 * ***fast():   Approximate and nonrobust.
 * ***exact():  Exact and robust.
 * ***slow():   Another exact and robust implementation.
 * ***():       Adaptive exact and robust.
 */

double orient2d(const double *pa, const double *pb, const double *pc);
double orient2dfast(const double *pa, const double *pb, const double *pc);
double orient2dexact(const double *pa, const double *pb, const double *pc);
double orient3d(const double *pa, const double *pb, const double *pc,
                const double *pd);
double orient3dfast(const double *pa, const double *pb, const double *pc,
                    const double *pd);
double orient3dexact(const double *pa, const double *pb, const double *pc,
                     const double *pd);
double incircle(const double *pa, const double *pb, const double *pc,
                const double *pd);
double incirclefast(const double *pa, const double *pb, const double *pc,
                    const double *pd);
double incircleexact(const double *pa, const double *pb, const double *pc,
                     const double *pd);
double insphere(const double *pa, const double *pb, const double *pc,
                const double *pd, const double *pe);
double inspherefast(const double *pa, const double *pb, const double *pc,
                    const double *pd, const double *pe);
double insphereexact(const double *pa, const double *pb, const double *pc,
                     const double *pd, const double *pe);
}

} // namespace Shewchuk

#endif // SHEWCHUK_H
