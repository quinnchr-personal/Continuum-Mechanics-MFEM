// Internal variables of a history-dependent material at the quadrature
// points of the kernels' rule (order 2p + 3 on each element's geometry):
// the accepted values, which are the state at the start of the step under
// way, a scratch buffer the update writes, and the physical length dt of the
// step under way. Between steps dt is 0, so a material evaluated then returns
// the stress of the accepted state. The layout is one block of `size`
// doubles per point, points of an element consecutive, elements in mesh
// order; a material that needs fewer than `size` values (a region with fewer
// branches) uses the head of its block.
#pragma once

#include <vector>

#include "mfem.hpp"

namespace cmf
{

class HistoryField
{
public:
  HistoryField(mfem::Mesh &mesh, int order, int size);

  int Size() const { return size_; }
  int Order() const { return order_; }
  int NumElements() const { return int(offsets_.size()) - 1; }
  const mfem::IntegrationRule &Rule(int elem) const
  {
    return mfem::IntRules.Get(mesh_.GetElementBaseGeometry(elem), order_);
  }
  int NumPoints(int elem) const { return offsets_[std::size_t(elem) + 1] - offsets_[std::size_t(elem)]; }

  // The accepted history at point q of element elem, and the block the
  // update of the step under way writes.
  const double *Old(int elem, int q) const { return old_.data() + Offset(elem, q); }
  double *New(int elem, int q) { return new_.data() + Offset(elem, q); }
  // Sets the accepted history of every point of elem to `values` (size doubles).
  void Fill(int elem, const double *values);
  // Makes the updated values the accepted ones.
  void Commit() { old_ = new_; }

  double Dt() const { return dt_; }
  void SetDt(double dt) { dt_ = dt; }

private:
  std::size_t Offset(int elem, int q) const
  {
    return std::size_t(offsets_[std::size_t(elem)] + q) * std::size_t(size_);
  }

  mfem::Mesh &mesh_;
  int order_;
  int size_;
  std::vector<int> offsets_; // first point of each element, then the total
  std::vector<double> old_, new_;
  double dt_ = 0.0;
};

} // namespace cmf
