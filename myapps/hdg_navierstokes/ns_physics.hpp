#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace hdg_ns
{

struct NSParams
{
   double mu[11] =
   {
      1.4, 183500.0, 0.71, 8.03, 1.0, 1.0, 0.0,
      0.52769, 0.027694, 124.49, 294.44
   };
   double tau = 1.0;

   double TisoW() const { return mu[10] / mu[9] * mu[8]; }
   double TinfFlux() const
   {
      return 1.0 / (mu[0] * (mu[0] - 1.0) * mu[3] * mu[3]);
   }
};

inline double Pressure(const double u[4], const NSParams &params)
{
   const double inv_r = 1.0 / u[0];
   return (params.mu[0] - 1.0) *
          (u[3] - 0.5 * inv_r * (u[1] * u[1] + u[2] * u[2]));
}

inline double SutherlandMu(double p, double r, const NSParams &params)
{
   const double gam = params.mu[0];
   const double re = params.mu[1];
   const double minf = params.mu[3];
   const double tref = params.mu[9];
   const double tr = gam * minf * minf * p / r;
   const double tphys = tref * tr;
   return std::pow(tr, 1.5) * (tref + 110.4) /
          (re * (tphys + 110.4));
}

namespace detail
{

struct TransportTerms
{
   double r;
   double ru;
   double rv;
   double rE;
   double uv;
   double vv;
   double p;
   double h;
   double muphys;
   double fc;
   double ux;
   double vx;
   double Tx;
   double uy;
   double vy;
   double Ty;
   double txx;
   double txy;
   double tyy;
};

inline TransportTerms ComputeTransport(const double uq[12],
                                       const NSParams &params)
{
   TransportTerms a{};
   a.r = uq[0];
   a.ru = uq[1];
   a.rv = uq[2];
   a.rE = uq[3];
   const double inv_r = 1.0 / a.r;
   const double gam = params.mu[0];
   const double gam1 = gam - 1.0;
   a.uv = a.ru * inv_r;
   a.vv = a.rv * inv_r;
   const double ke = 0.5 * (a.uv * a.uv + a.vv * a.vv);
   a.p = gam1 * (a.rE - a.r * ke);
   a.h = a.rE * inv_r + a.p * inv_r;
   a.muphys = SutherlandMu(a.p, a.r, params);
   a.fc = a.muphys * gam / params.mu[2];

   a.ux = (uq[5] - uq[4] * a.uv) * inv_r;
   a.vx = (uq[6] - uq[4] * a.vv) * inv_r;
   const double px = gam1 *
                     (uq[7] - uq[4] * ke -
                      a.r * (a.uv * a.ux + a.vv * a.vx));
   a.Tx = (px * a.r - a.p * uq[4]) /
          (gam1 * a.r * a.r);

   a.uy = (uq[9] - uq[8] * a.uv) * inv_r;
   a.vy = (uq[10] - uq[8] * a.vv) * inv_r;
   const double py = gam1 *
                     (uq[11] - uq[8] * ke -
                      a.r * (a.uv * a.uy + a.vv * a.vy));
   a.Ty = (py * a.r - a.p * uq[8]) /
          (gam1 * a.r * a.r);

   a.txx = a.muphys * (2.0 / 3.0) * (2.0 * a.ux - a.vy);
   a.txy = a.muphys * (a.uy + a.vx);
   a.tyy = a.muphys * (2.0 / 3.0) * (2.0 * a.vy - a.ux);
   return a;
}

// Mechanical transcription of nsmach8-baseline/my_model.hpp:222-531.
// Storage is dfduq[flux_component + 8 * uq_component].
inline void FluxJacobianGenerated(const double uq[12], double av,
                                  const NSParams &params, double f[96])
{
   const double uq0 = uq[0];
   const double uq1 = uq[1];
   const double uq2 = uq[2];
   const double uq3 = uq[3];
   const double uq4 = uq[4];
   const double uq5 = uq[5];
   const double uq6 = uq[6];
   const double uq7 = uq[7];
   const double uq8 = uq[8];
   const double uq9 = uq[9];
   const double uq10 = uq[10];
   const double uq11 = uq[11];
   const double v0 = av;
   const double mu0 = params.mu[0];
   const double mu1 = params.mu[1];
   const double mu2 = params.mu[2];
   const double mu3 = params.mu[3];
   const double mu9 = params.mu[9];

   const double x0 = std::pow(uq0, -1);
   const double x1 = x0*uq2;
   const double x2 = uq10 - x1*uq8;
   const double x3 = x0*x2;
   const double x4 = x0*uq1;
   const double x5 = uq5 - x4*uq4;
   const double x6 = x0*x5;
   const double x7 = -x3 + 2*x6;
   const double x8 = std::pow(uq1, 2);
   const double x9 = std::pow(uq0, -3);
   const double x10 = x8*x9;
   const double x11 = std::pow(uq2, 2);
   const double x12 = x9*x11;
   const double x13 = 0.5*(-2*x10 - 2*x12);
   const double x14 = std::pow(uq0, -2);
   const double x15 = x14*x11;
   const double x16 = x8*x14;
   const double x17 = -0.5*x15 - 0.5*x16;
   const double x18 = x17 - uq0*x13;
   const double x19 = 0.5*(x15 + x16);
   const double x20 = uq3 - uq0*x19;
   const double x21 = std::pow(x20, 2);
   const double x22 = std::pow(mu3, 6);
   const double x23 = -1.0 + mu0;
   const double x24 = x22*std::pow(x23, 3)*std::pow(mu0, 3);
   const double x25 = x24*x21;
   const double x26 = x9*x25;
   const double x27 = std::pow(uq0, -4);
   const double x28 = x24*std::pow(x20, 3);
   const double x29 = std::sqrt(x9*x28);
   const double x30 = std::pow(mu1, -1);
   const double x31 = 110.4 + mu9;
   const double x32 = x0*x23;
   const double x33 = x32*x20;
   const double x34 = 1.0*mu0;
   const double x35 = std::pow(mu3, 2)*mu9;
   const double x36 = x34*x35;
   const double x37 = 110.4 + x33*x36;
   const double x38 = x30*x31/x37;
   const double x39 = x38/x29;
   const double x40 = (3*x26*x18 - 3*x28*x27)*x39;
   const double x41 = 0.333333333333333*x40;
   const double x42 = x7*x41;
   const double x43 = x23*x20;
   const double x44 = x43*x14;
   const double x45 = x32*x18;
   const double x46 = x30*x31*x29/std::pow(x37, 2);
   const double x47 = (-x44*x36 + x45*x36)*x46;
   const double x48 = 0.666666666666667*x47;
   const double x49 = x7*x48;
   const double x50 = x23*x18;
   const double x51 = x5*x14;
   const double x52 = x9*uq4;
   const double x53 = uq1*x52;
   const double x54 = x2*x14;
   const double x55 = x9*uq8;
   const double x56 = uq2*x55;
   const double x57 = x54 - x56;
   const double x58 = x38*x29;
   const double x59 = 0.666666666666667*x58;
   const double x60 = x59*(-2*x51 + 2*x53 + x57);
   const double x61 = uq6 - x1*uq4;
   const double x62 = uq9 - x4*uq8;
   const double x63 = x0*x61 + x0*x62;
   const double x64 = 0.5*x40;
   const double x65 = x63*x64;
   const double x66 = x62*x14;
   const double x67 = uq1*x55;
   const double x68 = x61*x14;
   const double x69 = uq2*x52;
   const double x70 = 1.0*x58;
   const double x71 = (-x66 + x67 - x68 + x69)*x70;
   const double x72 = uq2*x14;
   const double x73 = uq1*x72;
   const double x74 = 1.0*x63*x47;
   const double x75 = x65 + x71 - x73 - x74;
   const double x76 = uq2*x68;
   const double x77 = uq1*x51;
   const double x78 = x23*(uq7 - uq4*x19 - (x76 + x77)*uq0);
   const double x79 = uq0*x78 - uq4*x43;
   const double x80 = x9*x58;
   const double x81 = std::pow(x23, -1);
   const double x82 = std::pow(mu2, -1);
   const double x83 = x82*mu0;
   const double x84 = x81*x83;
   const double x85 = 2.0*x80*x84;
   const double x86 = x82*x79;
   const double x87 = x81*x47*x34*x14;
   const double x88 = x84*x14;
   const double x89 = x88*x64;
   const double x90 = x70*x72;
   const double x91 = 2*x9;
   const double x92 = uq1*x91;
   const double x93 = uq2*x91;
   const double x94 = uq4*x27;
   const double x95 = uq0*x23;
   const double x96 = x88*x70;
   const double x97 = -x44 + x45 - uq3*x14;
   const double x98 = uq1*x14;
   const double x99 = x59*x98;
   const double x100 = 2*x3 - x6;
   const double x101 = x41*x100;
   const double x102 = x48*x100;
   const double x103 = x51 - x53;
   const double x104 = x59*(x103 - 2*x54 + 2*x56);
   const double x105 = uq2*x54;
   const double x106 = uq1*x66;
   const double x107 = x23*(uq11 - uq8*x19 - (x105 + x106)*uq0);
   const double x108 = uq0*x107 - uq8*x43;
   const double x109 = x82*x108;
   const double x110 = x70*x98;
   const double x111 = uq8*x27;
   const double x112 = x72*x59;
   const double x113 = 1.0*x32;
   const double x114 = uq1*x113;
   const double x115 = -x114;
   const double x116 = uq1*x27;
   const double x117 = x39*x25;
   const double x118 = 1.0*x117;
   const double x119 = x7*x118;
   const double x120 = x119*x116;
   const double x121 = 1.33333333333333*x58;
   const double x122 = uq4*x14;
   const double x123 = x46*x35;
   const double x124 = mu0*x123;
   const double x125 = 0.666666666666667*x124;
   const double x126 = x23*x125;
   const double x127 = x7*x126;
   const double x128 = x98*x127;
   const double x129 = 1.5*x63;
   const double x130 = x117*x129;
   const double x131 = x116*x130;
   const double x132 = 1.0*uq8;
   const double x133 = x58*x14;
   const double x134 = x133*x132;
   const double x135 = 1.0*x23;
   const double x136 = x63*x124;
   const double x137 = x135*x136;
   const double x138 = x98*x137;
   const double x139 = x1 - x131 - x134 + x138;
   const double x140 = std::pow(uq0, -5);
   const double x141 = x119*x140;
   const double x142 = 1.5*x39*x22*std::pow(x23, 2)*x21*std::pow(mu0, 4);
   const double x143 = x142/std::pow(uq0, 6);
   const double x144 = x86*x143;
   const double x145 = 1.0*std::pow(mu0, 2);
   const double x146 = x123*x145;
   const double x147 = x116*x146;
   const double x148 = x0*x59;
   const double x149 = 1.0*uq4;
   const double x150 = uq2*uq1;
   const double x151 = x130*x140;
   const double x152 = x9*x123;
   const double x153 = mu0*x150*x152;
   const double x154 = x33 + x0*uq3 - x150*x151 + x63*x135*x153;
   const double x155 = x100*x126;
   const double x156 = x100*x118;
   const double x157 = x109*x143;
   const double x158 = x140*x156;
   const double x159 = 0.666666666666667*x23*x153;
   const double x160 = x0*x70;
   const double x161 = x63*x160 - x73*x135;
   const double x162 = uq2*x113;
   const double x163 = -x162;
   const double x164 = uq2*x27;
   const double x165 = uq8*x14;
   const double x166 = x164*x130;
   const double x167 = x72*x137;
   const double x168 = x133*x149;
   const double x169 = -x166 + x167 - x168 + x4;
   const double x170 = x164*x146;
   const double x171 = x72*x155;
   const double x172 = x164*x156;
   const double x173 = x39*x26;
   const double x174 = 1.0*x173;
   const double x175 = x32*x125;
   const double x176 = -x113*x136 + x129*x173;
   const double x177 = x0 + x32;
   const double x178 = x145*x152;
   const double x179 = x140*x142;
   const double x180 = x98*x121;
   const double x181 = -x90;
   const double x182 = x96*(-x43 + x95*(x17 - (-x10 - x12)*uq0));
   const double x183 = -0.333333333333333*x80*x150;
   const double x184 = v0 + x0*x121;
   const double x185 = -x83*x110;
   const double x186 = -x148;
   const double x187 = v0 + x160;
   const double x188 = -x83*x90;
   const double x189 = v0 + x83*x160;
   const double x190 = -x110;
   const double x191 = x72*x121;

   f[0] = 0;
   f[1] = -x16 + x42 - x49 + x50 + x60;
   f[2] = x75;
   f[3] = uq1*x97 + x1*x65 + x1*x71 - x1*x74 + x4*x42 - x4*x49 + x4*x60 - x63*x90 - x7*x99 - x85*x79 - x86*x87 + x89*x79 + x96*(x78 - uq4*x50 + x95*(-x76 - x77 - uq4*x13 - (-x5*x92 - x61*x93 + x8*x94 + x94*x11)*uq0));
   f[4] = 0;
   f[5] = x75;
   f[6] = x101 - x102 + x104 - x15 + x50;
   f[7] = uq2*x97 + x1*x101 - x1*x102 + x1*x104 - x100*x112 + x4*x65 + x4*x71 - x4*x74 - x63*x110 - x85*x108 - x87*x109 + x89*x108 + x96*(x107 - uq8*x50 + x95*(-x105 - x106 - uq0*(x11*x111 - x2*x93 - x62*x92 + x8*x111) - uq8*x13));
   f[8] = 1;
   f[9] = x115 - x120 + x128 + 2*x4 - x122*x121;
   f[10] = x139;
   f[11] = x154 - uq1*x144 + x10*x127 - x16*x135 - x53*x121 + x7*x148 - x70*x56 - x8*x141 + x86*x147 + x96*(uq4*x114 + x95*(-uq0*x103 - x98*x149));
   f[12] = 0;
   f[13] = x139;
   f[14] = x115 - x116*x156 + x59*x122 + x98*x155;
   f[15] = x161 - uq1*x157 + x10*x137 + x100*x159 + x109*x147 - x150*x158 + x69*x59 - x70*x67 - x8*x151 + x96*(uq8*x114 + x95*(-x98*x132 - (x66 - x67)*uq0));
   f[16] = 0;
   f[17] = x163 - x119*x164 + x59*x165 + x72*x127;
   f[18] = x169;
   f[19] = x161 - uq2*x144 - x11*x151 + x12*x137 - x141*x150 + x67*x59 + x7*x159 - x70*x69 + x86*x170 + x96*(uq4*x162 + x95*(-x72*x149 - (x68 - x69)*uq0));
   f[20] = 1;
   f[21] = x169;
   f[22] = 2*x1 + x163 + x171 - x172 - x121*x165;
   f[23] = x154 - uq2*x157 + x100*x148 + x109*x170 - x11*x158 + x12*x155 - x15*x135 - x56*x121 - x70*x53 + x96*(uq8*x162 + x95*(-uq0*x57 - x72*x132));
   f[24] = 0;
   f[25] = x23 + x7*x174 - x7*x175;
   f[26] = x176;
   f[27] = x120 - x128 + x166 - x167 + uq1*x177 - x83*x168 - x86*x178 + x86*x179;
   f[28] = 0;
   f[29] = x176;
   f[30] = x23 + x100*x174 - x100*x175;
   f[31] = x131 - x138 - x171 + x172 + uq2*x177 - x109*x178 + x109*x179 - x83*x134;
   f[32] = v0;
   f[33] = -x180;
   f[34] = x181;
   f[35] = x182 - x10*x121 - x70*x12;
   f[36] = 0;
   f[37] = x181;
   f[38] = x99;
   f[39] = x183;
   f[40] = 0;
   f[41] = x184;
   f[42] = 0;
   f[43] = x180 + x185;
   f[44] = 0;
   f[45] = 0;
   f[46] = x186;
   f[47] = -x112;
   f[48] = 0;
   f[49] = 0;
   f[50] = x187;
   f[51] = x188 + x90;
   f[52] = 0;
   f[53] = x160;
   f[54] = 0;
   f[55] = x110;
   f[56] = 0;
   f[57] = 0;
   f[58] = 0;
   f[59] = x189;
   f[60] = 0;
   f[61] = 0;
   f[62] = 0;
   f[63] = 0;
   f[64] = 0;
   f[65] = x112;
   f[66] = x190;
   f[67] = x183;
   f[68] = v0;
   f[69] = x190;
   f[70] = -x191;
   f[71] = x182 - x12*x121 - x70*x10;
   f[72] = 0;
   f[73] = 0;
   f[74] = x160;
   f[75] = x90;
   f[76] = 0;
   f[77] = x187;
   f[78] = 0;
   f[79] = x110 + x185;
   f[80] = 0;
   f[81] = x186;
   f[82] = 0;
   f[83] = -x99;
   f[84] = 0;
   f[85] = 0;
   f[86] = x184;
   f[87] = x188 + x191;
   f[88] = 0;
   f[89] = 0;
   f[90] = 0;
   f[91] = 0;
   f[92] = 0;
   f[93] = 0;
   f[94] = 0;
   f[95] = x189;
}

} // namespace detail

inline void NSFlux(const double uq[12], double av, const NSParams &params,
                   double f[8], double *dfduq = nullptr)
{
   const detail::TransportTerms a = detail::ComputeTransport(uq, params);
   f[0] = a.ru + av * uq[4];
   f[1] = a.ru * a.uv + a.p + a.txx + av * uq[5];
   f[2] = a.rv * a.uv + a.txy + av * uq[6];
   f[3] = a.ru * a.h + a.uv * a.txx + a.vv * a.txy +
          a.fc * a.Tx + av * uq[7];
   f[4] = a.rv + av * uq[8];
   f[5] = a.ru * a.vv + a.txy + av * uq[9];
   f[6] = a.rv * a.vv + a.p + a.tyy + av * uq[10];
   f[7] = a.rv * a.h + a.uv * a.txy + a.vv * a.tyy +
          a.fc * a.Ty + av * uq[11];
   if (dfduq)
   {
      detail::FluxJacobianGenerated(uq, av, params, dfduq);
   }
}

inline void NSFbouHdg(int ib, const double uq[12], const double uhat[4],
                      const double[2], const NSParams &params, double fb[4],
                      double *dfbduq = nullptr, double *dfbduh = nullptr)
{
   if (ib < 1 || ib > 3)
   {
      throw std::invalid_argument("NSFbouHdg: ib must be 1, 2, or 3");
   }
   std::fill(fb, fb + 4, 0.0);
   if (dfbduq) { std::fill(dfbduq, dfbduq + 48, 0.0); }
   if (dfbduh) { std::fill(dfbduh, dfbduh + 16, 0.0); }

   if (ib == 1)
   {
      for (int c = 0; c < 4; ++c)
      {
         fb[c] = params.mu[4 + c] - uhat[c];
         if (dfbduh) { dfbduh[c + 4 * c] = -1.0; }
      }
   }
   else if (ib == 2)
   {
      for (int c = 0; c < 4; ++c)
      {
         fb[c] = uq[c] - uhat[c];
         if (dfbduq) { dfbduq[c + 4 * c] = 1.0; }
         if (dfbduh) { dfbduh[c + 4 * c] = -1.0; }
      }
   }
   else
   {
      fb[0] = uq[0] - uhat[0];
      fb[1] = -uhat[1];
      fb[2] = -uhat[2];
      fb[3] = params.TisoW() * uhat[0] - uhat[3];
      if (dfbduq) { dfbduq[0] = 1.0; }
      if (dfbduh)
      {
         dfbduh[0] = -1.0;
         dfbduh[5] = -1.0;
         dfbduh[10] = -1.0;
         dfbduh[3] = params.TisoW();
         dfbduh[15] = -1.0;
      }
   }
}

inline void NSHeatFlux(const double uhat[4], const double uq[12],
                       const NSParams &params, double f[2])
{
   double work[12];
   std::copy(uhat, uhat + 4, work);
   std::copy(uq + 4, uq + 12, work + 4);
   const detail::TransportTerms a = detail::ComputeTransport(work, params);
   f[0] = a.fc * a.Tx;
   f[1] = a.fc * a.Ty;
}

inline void NSVisScalars(const double u[4], double s[4])
{
   const double inv_r = 1.0 / u[0];
   s[0] = u[0];
   s[1] = u[1] * inv_r;
   s[2] = u[2] * inv_r;
   s[3] = 0.4 *
          (u[3] - 0.5 * inv_r * (u[1] * u[1] + u[2] * u[2]));
}

inline void FluxPhysGrad(const double uq_phys[12], double av,
                         const NSParams &params, double f[8],
                         double *dfduq_phys = nullptr)
{
   double uq_model[12];
   std::copy(uq_phys, uq_phys + 4, uq_model);
   for (int i = 4; i < 12; ++i) { uq_model[i] = -uq_phys[i]; }
   NSFlux(uq_model, av, params, f, dfduq_phys);
   if (dfduq_phys)
   {
      for (int variable = 4; variable < 12; ++variable)
      {
         for (int output = 0; output < 8; ++output)
         {
            dfduq_phys[output + 8 * variable] *= -1.0;
         }
      }
   }
}

} // namespace hdg_ns
