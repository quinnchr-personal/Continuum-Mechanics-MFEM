#include "exasim_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hdg_ns
{
namespace
{

int CheckedInteger(double value, const std::string &path,
                   const std::string &label)
{
   if (!std::isfinite(value) || value < 0.0 ||
       value != std::round(value) ||
       value > static_cast<double>(std::numeric_limits<int>::max()))
   {
      std::ostringstream message;
      message << path << ": invalid integer-valued " << label
              << " entry " << value;
      throw std::runtime_error(message.str());
   }
   return static_cast<int>(value);
}

std::vector<int> ReadIntegers(const std::vector<double> &raw,
                              std::size_t &cursor, int count,
                              const std::string &path,
                              const std::string &label)
{
   if (count < 0 || cursor + static_cast<std::size_t>(count) > raw.size())
   {
      throw std::runtime_error(path + ": truncated " + label);
   }
   std::vector<int> result(static_cast<std::size_t>(count));
   for (int i = 0; i < count; ++i)
   {
      result[static_cast<std::size_t>(i)] =
         CheckedInteger(raw[cursor++], path, label);
   }
   return result;
}

struct RankInput
{
   std::vector<int> elempart;
   std::vector<int> elempartpts;
   std::vector<int> perm;
   std::vector<int> bf;
   std::vector<int> ti;
   ExasimArray xdg;
   ExasimArray outudg;
   ExasimArray outuhat;
};

RankInput ReadRankInput(const std::string &run_directory, int rank)
{
   RankInput input;
   const std::string rank_string = std::to_string(rank + 1);
   const std::string mesh_path =
      run_directory + "/datain/mesh" + rank_string + ".bin";
   const std::vector<double> raw = ReadExasimDoubles(mesh_path);
   std::size_t cursor = 0;
   if (raw.empty())
   {
      throw std::runtime_error(mesh_path + ": empty mesh file");
   }
   const int nsize_count =
      CheckedInteger(raw[cursor++], mesh_path, "nsize length");
   const std::vector<int> nsize =
      ReadIntegers(raw, cursor, nsize_count, mesh_path, "nsize");
   if (nsize.size() < 29)
   {
      throw std::runtime_error(mesh_path + ": nsize is shorter than 29");
   }
   const std::vector<int> ndims =
      ReadIntegers(raw, cursor, nsize[0], mesh_path, "ndims");
   if (ndims.size() < 5 || ndims[0] != 2 || ndims[4] != 4)
   {
      throw std::runtime_error(mesh_path + ": expected a 2D quad mesh");
   }

   ReadIntegers(raw, cursor, nsize[1], mesh_path, "facecon");
   ReadIntegers(raw, cursor, nsize[2], mesh_path, "eblks");
   ReadIntegers(raw, cursor, nsize[3], mesh_path, "fblks");
   ReadIntegers(raw, cursor, nsize[4], mesh_path, "nbsd");
   ReadIntegers(raw, cursor, nsize[5], mesh_path, "elemsend");
   ReadIntegers(raw, cursor, nsize[6], mesh_path, "elemrecv");
   ReadIntegers(raw, cursor, nsize[7], mesh_path, "elemsendpts");
   ReadIntegers(raw, cursor, nsize[8], mesh_path, "elemrecvpts");
   input.elempart =
      ReadIntegers(raw, cursor, nsize[9], mesh_path, "elempart");
   input.elempartpts =
      ReadIntegers(raw, cursor, nsize[10], mesh_path, "elempartpts");
   input.perm =
      ReadIntegers(raw, cursor, nsize[23], mesh_path, "perm");
   input.bf =
      ReadIntegers(raw, cursor, nsize[24], mesh_path, "bf");
   ReadIntegers(raw, cursor, nsize[25], mesh_path, "cartGridPart");
   input.ti =
      ReadIntegers(raw, cursor, nsize[26], mesh_path, "ti");
   ReadIntegers(raw, cursor, nsize[27], mesh_path,
                "boundaryConditions");
   ReadIntegers(raw, cursor, nsize[28], mesh_path, "intepartpts");
   if (cursor != raw.size())
   {
      throw std::runtime_error(mesh_path + ": unparsed mesh payload");
   }
   if (input.elempartpts.size() < 3 ||
       input.perm.size() != 20 ||
       input.bf.size() != 4 * input.elempart.size() ||
       input.ti.size() != 4 * input.elempart.size())
   {
      throw std::runtime_error(mesh_path + ": inconsistent HDG mesh arrays");
   }

   const std::string sol_path =
      run_directory + "/datain/sol" + rank_string + ".bin";
   const std::vector<double> sol_raw = ReadExasimDoubles(sol_path);
   cursor = 0;
   const int sol_nsize_count =
      CheckedInteger(sol_raw.at(cursor++), sol_path, "nsize length");
   const std::vector<int> sol_nsize =
      ReadIntegers(sol_raw, cursor, sol_nsize_count, sol_path, "nsize");
   if (sol_nsize.size() < 6)
   {
      throw std::runtime_error(sol_path + ": nsize is shorter than 6");
   }
   const std::vector<int> sol_ndims =
      ReadIntegers(sol_raw, cursor, sol_nsize[0], sol_path, "ndims");
   if (sol_ndims.size() < 12 || sol_ndims[3] != 25 ||
       sol_ndims[11] != 2 ||
       sol_ndims[0] != static_cast<int>(input.elempart.size()))
   {
      throw std::runtime_error(sol_path + ": unexpected solution metadata");
   }
   const int local_elements = static_cast<int>(input.elempart.size());
   if (sol_nsize[1] != 25 * 2 * local_elements)
   {
      throw std::runtime_error(sol_path + ": rank xdg has wrong size");
   }
   input.xdg.nnode = 25;
   input.xdg.ncomp = 2;
   input.xdg.nelem = local_elements;
   input.xdg.data.assign(
      sol_raw.begin() + static_cast<std::ptrdiff_t>(cursor),
      sol_raw.begin() + static_cast<std::ptrdiff_t>(
         cursor + static_cast<std::size_t>(sol_nsize[1])));

   input.outudg = ReadExasimArray(
      run_directory + "/dataout/outudg_np" +
      std::to_string(rank) + ".bin");
   const ExasimArray raw_outuhat = ReadExasimArray(
      run_directory + "/dataout/outuhat_np" +
      std::to_string(rank) + ".bin");
   if (raw_outuhat.nnode != 4 || raw_outuhat.ncomp != 5)
   {
      throw std::runtime_error(
         "outuhat header is not Exasim's [ncu,npf,nf]=[4,5,nf]");
   }
   // solutionwriter records sol.uh as [ncu,npf,nf], component-fast.
   // Normalize it to ExasimArray's [node,component,element] contract.
   input.outuhat.nnode = 5;
   input.outuhat.ncomp = 4;
   input.outuhat.nelem = raw_outuhat.nelem;
   input.outuhat.data.resize(raw_outuhat.data.size());
   for (int face = 0; face < input.outuhat.nelem; ++face)
   {
      for (int component = 0; component < 4; ++component)
      {
         for (int node = 0; node < 5; ++node)
         {
            input.outuhat(node, component, face) =
               raw_outuhat.data[
                  static_cast<std::size_t>(component) +
                  4 * (static_cast<std::size_t>(node) +
                       5 * static_cast<std::size_t>(face))];
         }
      }
   }
   return input;
}

struct FaceKey
{
   std::array<int, 2> vertices{};

   bool operator==(const FaceKey &other) const
   {
      return vertices == other.vertices;
   }
};

struct FaceKeyHash
{
   std::size_t operator()(const FaceKey &key) const noexcept
   {
      std::uint64_t hash = 0x9E3779B97F4A7C15ull ^ 2u;
      for (int vertex : key.vertices)
      {
         std::uint64_t value =
            static_cast<std::uint64_t>(vertex) +
            0x9E3779B97F4A7C15ull;
         value ^= value >> 30;
         value *= 0xBF58476D1CE4E5B9ull;
         value ^= value >> 27;
         value *= 0x94D049BB133111EBull;
         value ^= value >> 31;
         hash ^= value + 0x9E3779B97F4A7C15ull +
                 (hash << 6) + (hash >> 2);
      }
      return static_cast<std::size_t>(hash);
   }
};

struct LocalFace
{
   int element1 = -1;
   int local_face1 = -1;
   int element2 = -1;
   int local_face2 = -1;
};

constexpr std::array<std::array<int, 2>, 4> kLocalFaceVertices =
{{
   {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}}
}};

FaceKey MakeFaceKey(const RankInput &rank, int element, int local_face)
{
   FaceKey key;
   key.vertices[0] =
      rank.ti[kLocalFaceVertices[local_face][0] + 4 * element];
   key.vertices[1] =
      rank.ti[kLocalFaceVertices[local_face][1] + 4 * element];
   std::sort(key.vertices.begin(), key.vertices.end());
   return key;
}

std::vector<LocalFace> BuildRankFaces(const RankInput &rank)
{
   using Owner = std::pair<int, int>;
   std::unordered_map<FaceKey, Owner, FaceKeyHash> unmatched;
   const int ne = static_cast<int>(rank.elempart.size());
   unmatched.reserve(
      static_cast<std::size_t>(
         static_cast<double>(static_cast<std::size_t>(ne) * 4) * 1.3));
   std::vector<LocalFace> raw_faces;
   raw_faces.reserve(static_cast<std::size_t>(4 * ne));
   for (int element = 0; element < ne; ++element)
   {
      for (int local_face = 0; local_face < 4; ++local_face)
      {
         const FaceKey key = MakeFaceKey(rank, element, local_face);
         const auto match = unmatched.find(key);
         if (match == unmatched.end())
         {
            unmatched.emplace(key, Owner{element, local_face});
         }
         else
         {
            raw_faces.push_back(
               {match->second.first, match->second.second,
                element, local_face});
            unmatched.erase(match);
         }
      }
   }
   for (const auto &entry : unmatched)
   {
      raw_faces.push_back(
         {entry.second.first, entry.second.second, -1, -1});
   }

   const int ne1 = rank.elempartpts[0] + rank.elempartpts[1];
   std::vector<LocalFace> interior, boundary;
   for (const LocalFace &face : raw_faces)
   {
      if (face.element1 >= ne1) { continue; }
      if (face.element2 >= 0) { interior.push_back(face); }
      else { boundary.push_back(face); }
   }
   std::vector<int> boundary_order(boundary.size());
   std::vector<int> boundary_attributes(boundary.size());
   for (std::size_t i = 0; i < boundary.size(); ++i)
   {
      boundary_order[i] = static_cast<int>(i);
      boundary_attributes[i] =
         rank.bf[boundary[i].local_face1 + 4 * boundary[i].element1];
   }
   for (std::size_t i = 0; i + 1 < boundary.size(); ++i)
   {
      for (std::size_t j = 0; j + i + 1 < boundary.size(); ++j)
      {
         if (boundary_attributes[j] > boundary_attributes[j + 1])
         {
            std::swap(boundary_attributes[j],
                      boundary_attributes[j + 1]);
            std::swap(boundary_order[j], boundary_order[j + 1]);
         }
      }
   }

   std::vector<LocalFace> faces;
   faces.reserve(interior.size() + boundary.size());
   faces.insert(faces.end(), interior.begin(), interior.end());
   for (int index : boundary_order)
   {
      faces.push_back(boundary[static_cast<std::size_t>(index)]);
   }
   for (LocalFace &face : faces)
   {
      if (face.element2 >= 0 &&
          rank.elempart[face.element2] <
          rank.elempart[face.element1])
      {
         std::swap(face.element1, face.element2);
         std::swap(face.local_face1, face.local_face2);
      }
   }
   return faces;
}

std::map<std::pair<int, int>, int> BuildGlobalFaceMap(
   const mfem::Mesh &mesh)
{
   std::map<std::pair<int, int>, int> result;
   mfem::Array<int> vertices;
   for (int face = 0; face < mesh.GetNumFaces(); ++face)
   {
      mesh.GetFaceVertices(face, vertices);
      if (vertices.Size() != 2)
      {
         throw std::runtime_error("reference face map expected segments");
      }
      const auto key = std::minmax(vertices[0], vertices[1]);
      if (!result.emplace(key, face).second)
      {
         throw std::runtime_error("duplicate global mesh edge");
      }
   }
   return result;
}

double RelativeDifference(double first, double second)
{
   return std::abs(first - second) /
          std::max({1.0, std::abs(first), std::abs(second)});
}

} // namespace

ExasimReferenceData ReadExasimReferenceData(
   const std::string &run_directory, const ExasimMesh &converted)
{
   if (!converted.mesh)
   {
      throw std::runtime_error("reference reader received a null mesh");
   }
   ExasimReferenceData reference;
   reference.udg.nnode = 25;
   reference.udg.ncomp = 12;
   reference.udg.nelem = converted.mesh->GetNE();
   reference.udg.data.assign(
      static_cast<std::size_t>(25 * 12 * reference.udg.nelem),
      std::numeric_limits<double>::quiet_NaN());
   reference.trace_face_values.resize(
      static_cast<std::size_t>(converted.mesh->GetNumFaces()));
   std::vector<char> element_assigned(
      static_cast<std::size_t>(converted.mesh->GetNE()), 0);
   std::vector<char> face_assigned(
      static_cast<std::size_t>(converted.mesh->GetNumFaces()), 0);
   const auto global_faces = BuildGlobalFaceMap(*converted.mesh);
   mfem::DG_Interface_FECollection trace_collection(kOrder, 2);
   const mfem::FiniteElement *trace_element =
      trace_collection.FiniteElementForGeometry(mfem::Geometry::SEGMENT);
   if (!trace_element || trace_element->GetDof() != 5)
   {
      throw std::runtime_error("MFEM did not provide the Q4 trace element");
   }
   const mfem::IntegrationRule &target_nodes = trace_element->GetNodes();
   mfem::Vector physical(2);

   for (int rank_number = 0; rank_number < 2; ++rank_number)
   {
      const RankInput rank =
         ReadRankInput(run_directory, rank_number);
      const int owned =
         rank.elempartpts[0] + rank.elempartpts[1];
      if (rank.outudg.nnode != 25 || rank.outudg.ncomp != 12 ||
          rank.outudg.nelem != owned ||
          rank.outuhat.nnode != 5 || rank.outuhat.ncomp != 4)
      {
         std::ostringstream message;
         message << "rank " << rank_number
                 << " output header does not match the M3 reference layout:"
                 << " elempartpts=" << rank.elempartpts[0] << ','
                 << rank.elempartpts[1] << ','
                 << rank.elempartpts[2]
                 << " outudg=[" << rank.outudg.nnode << ','
                 << rank.outudg.ncomp << ',' << rank.outudg.nelem << ']'
                 << " outuhat=[" << rank.outuhat.nnode << ','
                 << rank.outuhat.ncomp << ',' << rank.outuhat.nelem << ']';
         throw std::runtime_error(message.str());
      }

      for (int local_element = 0;
           local_element < static_cast<int>(rank.elempart.size());
           ++local_element)
      {
         const int global_element = rank.elempart[local_element];
         if (global_element < 0 ||
             global_element >= converted.xdg.nelem)
         {
            throw std::runtime_error("elempart contains an invalid element");
         }
         for (int component = 0; component < 2; ++component)
         {
            for (int node = 0; node < 25; ++node)
            {
               const double local =
                  rank.xdg(node, component, local_element);
               const double global =
                  converted.xdg(node, component, global_element);
               reference.maximum_xdg_difference =
                  std::max(reference.maximum_xdg_difference,
                           std::abs(local - global));
               if (std::memcmp(&local, &global, sizeof(double)) != 0)
               {
                  std::ostringstream message;
                  message << "rank " << rank_number
                          << " xdg byte mismatch at local element "
                          << local_element << ", global element "
                          << global_element << ", component " << component
                          << ", node " << node;
                  throw std::runtime_error(message.str());
               }
            }
         }
      }
      for (int local_element = 0; local_element < owned; ++local_element)
      {
         const int global_element = rank.elempart[local_element];
         if (element_assigned[static_cast<std::size_t>(global_element)])
         {
            throw std::runtime_error("owned element appears on two ranks");
         }
         element_assigned[static_cast<std::size_t>(global_element)] = 1;
         for (int component = 0; component < 12; ++component)
         {
            for (int node = 0; node < 25; ++node)
            {
               reference.udg(node, component, global_element) =
                  rank.outudg(node, component, local_element);
            }
         }
      }

      const std::vector<LocalFace> local_faces = BuildRankFaces(rank);
      if (static_cast<int>(local_faces.size()) != rank.outuhat.nelem)
      {
         std::ostringstream message;
         message << "rank " << rank_number << " reconstructed "
                 << local_faces.size() << " faces, outuhat contains "
                 << rank.outuhat.nelem;
         throw std::runtime_error(message.str());
      }
      reference.local_face_count += static_cast<int>(local_faces.size());
      for (int local_face_number = 0;
           local_face_number < static_cast<int>(local_faces.size());
           ++local_face_number)
      {
         const LocalFace &local_face = local_faces[local_face_number];
         const FaceKey key =
            MakeFaceKey(rank, local_face.element1,
                        local_face.local_face1);
         const auto global_match =
            global_faces.find(
               {key.vertices[0], key.vertices[1]});
         if (global_match == global_faces.end())
         {
            throw std::runtime_error(
               "rank face does not exist in the global mesh");
         }
         const int global_face = global_match->second;
         const int global_element =
            rank.elempart[local_face.element1];
         const int first_volume_node =
            rank.perm[5 * local_face.local_face1];
         const int last_volume_node =
            rank.perm[4 + 5 * local_face.local_face1];
         const std::array<double, 2> first =
         {{
            converted.xdg(first_volume_node, 0, global_element),
            converted.xdg(first_volume_node, 1, global_element)
         }};
         const std::array<double, 2> last =
         {{
            converted.xdg(last_volume_node, 0, global_element),
            converted.xdg(last_volume_node, 1, global_element)
         }};
         mfem::FaceElementTransformations *transformation =
            converted.mesh->GetFaceElementTransformations(global_face, 31);
         mfem::IntegrationPoint endpoint;
         endpoint.x = 0.0;
         transformation->Face->Transform(endpoint, physical);
         const double same_distance =
            std::hypot(physical[0] - first[0],
                       physical[1] - first[1]);
         const double reverse_distance =
            std::hypot(physical[0] - last[0],
                       physical[1] - last[1]);
         const bool same_direction = same_distance <= reverse_distance;

         for (int node = 0; node < 5; ++node)
         {
            endpoint.x = same_direction ?
               ExasimNodes1D()[node] : 1.0 - ExasimNodes1D()[node];
            transformation->Face->Transform(endpoint, physical);
            const int volume_node =
               rank.perm[node + 5 * local_face.local_face1];
            for (int component = 0; component < 2; ++component)
            {
               reference.maximum_face_geometry_difference =
                  std::max(
                     reference.maximum_face_geometry_difference,
                     RelativeDifference(
                        physical[component],
                        converted.xdg(volume_node, component,
                                      global_element)));
            }
         }
         if (reference.maximum_face_geometry_difference > 1.0e-13)
         {
            throw std::runtime_error(
               "rank outuhat face map failed geometric validation");
         }

         std::array<double, 20> target{};
         for (int dof = 0; dof < 5; ++dof)
         {
            const double target_coordinate =
               target_nodes.IntPoint(dof).x;
            const double source_coordinate = same_direction ?
               target_coordinate : 1.0 - target_coordinate;
            const auto shape = Lagrange1D(source_coordinate);
            for (int component = 0; component < 4; ++component)
            {
               for (int node = 0; node < 5; ++node)
               {
                  target[component + 4 * dof] +=
                     shape[node] *
                     rank.outuhat(node, component, local_face_number);
               }
            }
         }
         if (face_assigned[static_cast<std::size_t>(global_face)])
         {
            ++reference.duplicate_face_count;
            const auto &existing =
               reference.trace_face_values[
                  static_cast<std::size_t>(global_face)];
            for (int i = 0; i < 20; ++i)
            {
               reference.maximum_shared_trace_difference =
                  std::max(
                     reference.maximum_shared_trace_difference,
                     RelativeDifference(existing[i], target[i]));
            }
         }
         else
         {
            face_assigned[static_cast<std::size_t>(global_face)] = 1;
            reference.trace_face_values[
               static_cast<std::size_t>(global_face)] = target;
         }
      }
   }

   if (std::find(element_assigned.begin(), element_assigned.end(), 0) !=
       element_assigned.end() ||
       std::find(face_assigned.begin(), face_assigned.end(), 0) !=
       face_assigned.end())
   {
      throw std::runtime_error(
         "reference reassembly did not cover every element and face");
   }
   if (reference.local_face_count != 1377 ||
       reference.duplicate_face_count != 23 ||
       reference.maximum_shared_trace_difference > 1.0e-12)
   {
      throw std::runtime_error(
         "rank outuhat maps fail duplicate-face validation");
   }
   return reference;
}

void LoadExasimReferenceState(const ExasimReferenceData &reference,
                              HDGNavierStokesOperator &op,
                              bool load_gradient,
                              HDGState &state)
{
   op.LoadExasimVolumeState(reference.udg, load_gradient, state);
   if (reference.trace_face_values.size() * 20 !=
       static_cast<std::size_t>(op.TraceVSize()))
   {
      throw std::runtime_error("reference trace state has the wrong size");
   }
   for (int face = 0;
        face < static_cast<int>(reference.trace_face_values.size());
        ++face)
   {
      op.SetTraceFaceState(
         face, reference.trace_face_values[static_cast<std::size_t>(face)],
         state);
   }
   if (!load_gradient)
   {
      op.RecomputeGradient(state);
   }
}

} // namespace hdg_ns
