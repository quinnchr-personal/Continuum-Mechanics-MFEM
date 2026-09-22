#include "kernels/history_field.hpp"

#include <algorithm>

namespace cmf
{

HistoryField::HistoryField(mfem::Mesh &mesh, int order, int size)
  : mesh_(mesh), order_(order), size_(size)
{
  offsets_.assign(std::size_t(mesh.GetNE()) + 1, 0);
  for (int e = 0; e < mesh.GetNE(); e++)
  {
    offsets_[std::size_t(e) + 1] = offsets_[std::size_t(e)] + Rule(e).GetNPoints();
  }
  old_.assign(std::size_t(offsets_.back()) * std::size_t(size_), 0.0);
  new_ = old_;
}

void HistoryField::Fill(int elem, const double *values)
{
  for (int q = 0; q < NumPoints(elem); q++)
  {
    std::copy(values, values + size_, old_.data() + Offset(elem, q));
    std::copy(values, values + size_, new_.data() + Offset(elem, q));
  }
}

} // namespace cmf
