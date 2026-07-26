#!/usr/bin/env python3
"""Write udg.bin / vdg.bin for one continuation stage of nsmach8.

Replicates pdeapp_ns.m lines 55-67 exactly:
  dist = wall distance (= r - 1 for this mesh, verified to 2.8e-17)
  vdg comp0 = lambda*tanh(c*dist), comp1 = 0
  IC: u = [1, tanh(10 d), 0, TnearWall + 0.5 tanh(10 d)^2],
      TnearWall = Tinf*((Twall/Tref - 1) exp(-10 d) + 1),
      Tinf = 1/(gam*(gam-1)*Minf^2) full precision, q slots = 0.

Usage:
  write_stage.py --dir D --lambda L --c C [--ic]          # IC from scratch
  write_stage.py --dir D --lambda L --c C --udg-from F    # restart from F
"""
import argparse
import numpy as np


def read_bin(path):
    raw = np.fromfile(path, dtype=np.float64)
    n0, n1, n2 = (int(v) for v in raw[:3])
    data = raw[3:].reshape((n2, n1, n0)).transpose(2, 1, 0)  # (node, comp, elem)
    return data


def write_bin(path, array):
    n0, n1, n2 = array.shape
    header = np.array([n0, n1, n2], dtype=np.float64)
    with open(path, "wb") as handle:
        header.tofile(handle)
        array.transpose(2, 1, 0).astype(np.float64).tofile(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--c", type=float, required=True)
    parser.add_argument("--ic", action="store_true")
    parser.add_argument("--udg-from", default=None)
    args = parser.parse_args()

    xdg = read_bin(f"{args.dir}/xdg.bin")           # (25, 2, 651)
    npe, _, ne = xdg.shape
    dist = np.hypot(xdg[:, 0, :], xdg[:, 1, :]) - 1.0

    vdg = np.zeros((npe, 2, ne))
    vdg[:, 0, :] = args.lam * np.tanh(args.c * dist)
    write_bin(f"{args.dir}/vdg.bin", vdg)

    if args.ic:
        gam, Minf = 1.4, 8.03
        Tref, Twall = 124.49, 294.44
        Tinf = 1.0 / (gam * (gam - 1.0) * Minf * Minf)
        udg = np.zeros((npe, 12, ne))
        udg[:, 0, :] = 1.0
        udg[:, 1, :] = np.tanh(10.0 * dist)
        udg[:, 2, :] = 0.0
        t_near_wall = Tinf * ((Twall / Tref - 1.0) * np.exp(-10.0 * dist) + 1.0)
        udg[:, 3, :] = t_near_wall + 0.5 * udg[:, 1, :] ** 2
        write_bin(f"{args.dir}/udg.bin", udg)
    elif args.udg_from is not None:
        udg = read_bin(args.udg_from)
        assert udg.shape[1] == 12, udg.shape
        write_bin(f"{args.dir}/udg.bin", udg)

    print(f"stage files written: lambda={args.lam} c={args.c} "
          f"ic={args.ic} udg_from={args.udg_from}")


if __name__ == "__main__":
    main()
