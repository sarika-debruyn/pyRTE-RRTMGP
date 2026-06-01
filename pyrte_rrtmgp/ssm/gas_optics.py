import numpy as np
import xarray as xr

from .defaults import MOL_WEIGHTS
from .kernels import (
    compute_absorption_coeffs,
    compute_layer_mass,
    compute_planck_source,
    compute_tau,
)

class GasOptics:
    def __init__(
        self,
        optics_data: xr.Dataset,
        nus: xr.DataArray,
        dnus: xr.DataArray,
        pref=1.0e5,
    ):
        self.optics_data = optics_data
        self.triangles = optics_data["triangles"].rename(
            {"tags": "tag", "params": "param"}
        )

        self.nus = nus
        self.dnus = dnus
        self.pref = float(pref)

        self.tags = tuple(str(tag) for tag in self.triangles.coords["tag"].values)

        self.gases_by_tag = xr.DataArray(
            [tag.split("-")[0] for tag in self.tags],
            dims=("tag",),
            coords={"tag": self.tags},
            name="gas",
        )

        self.mol_weights_by_tag = xr.DataArray(
            [MOL_WEIGHTS[str(gas)] for gas in self.gases_by_tag.values],
            dims=("tag",),
            coords={
                "tag": self.tags,
                "gas": ("tag", self.gases_by_tag.values),
            },
            name="mol_weights",
            attrs={"units": "kg mol^-1"},
        )

        self._validate_inputs()

        self.absorption_coeffs = compute_absorption_coeffs(
            triangles=self.triangles,
            nus=self.nus,
        )
    #delete this if all input data are already set up in ideal shape
    def _init_inputs(self, atmos_data, nus, nu_min, nu_max, pref):
        self.tags = tuple(str(tag).lower() for tag in atmos_data.coords["tags"].values)
        self.gases_by_tag = xr.DataArray(
            [tag.split("-")[0] for tag in self.tags],
            dims=("tag",),
            coords={"tag": self.tags},
            name="gas",
        )
        self.gases = tuple(dict.fromkeys(self.gases_by_tag.values))

        self.triangles = atmos_data["triangles"].rename(
            {"tags": "tag", "params": "param"}
        )
        self.triangles = self.triangles.assign_coords(
            tag=self.tags,
            gas=("tag", self.gases_by_tag.values),
        )

        self.nus = xr.DataArray(
            nus,
            dims=("gpt",),
            name="nus",
            attrs={"units": "cm^-1"},
        )

        self.nu_min = float(nu_min)
        self.nu_max = float(nu_max)
        self.pref = float(pref)

        self.mol_weights_by_tag = xr.DataArray(
            [MOL_WEIGHTS[gas] for gas in self.gases_by_tag.values],
            dims=("tag",),
            coords={
                "tag": self.tags,
                "gas": ("tag", self.gases_by_tag.values),
            },
            name="mol_weights",
            attrs={"units": "kg mol^-1"},
        )

    def compute(self, atmos: xr.Dataset) -> xr.Dataset:
        play = atmos["play"]
        plev = atmos["plev"]
        tlay = atmos["tlay"]
        tlev = atmos["tlev"]
        tsfc = atmos["tsfc"]

        vmr = xr.concat(
            [atmos[gas] for gas in self.gases_by_tag.values],
            dim=xr.IndexVariable("tag", self.tags),
        )

        vmr = vmr.assign_coords(
            gas=("tag", self.gases_by_tag.values),
        )

        layer_mass = compute_layer_mass(
            vmr=vmr,
            plev=plev,
            play=play,
            mol_weights=self.mol_weights_by_tag,
        )

        tau = compute_tau(
            absorption_coeffs=self.absorption_coeffs,
            play=play,
            pref=self.pref,
            layer_mass=layer_mass,
        )

        return atmos.assign(
            tau=tau,
            lay_source=compute_planck_source(tlay, self.nus, self.dnus),
            lev_source=compute_planck_source(tlev, self.nus, self.dnus),
            sfc_source=compute_planck_source(tsfc, self.nus, self.dnus),
            nus=self.nus,
            dnus=self.dnus,
        )

    def _validate_inputs(self):
        if len(self.tags) == 0:
            raise ValueError("optics_data must contain at least one tag")

        if len(set(self.tags)) != len(self.tags):
            raise ValueError("tags must be unique")

        for gas in self.gases:
            if gas not in MOL_WEIGHTS:
                raise ValueError(f"Unknown gas name: {gas}")

        if self.triangles.ndim != 2:
            raise ValueError("triangles must be 2D with dimensions tag and param")

        if set(self.triangles.dims) != {"tag", "param"}:
            raise ValueError("triangles must have dimensions tag and param")

        required_params = {"nu0", "l", "kappa0"}
        params = set(str(p) for p in self.triangles.coords["param"].values)

        if params != required_params:
            raise ValueError("triangles params must be exactly nu0, l, and kappa0")

        if not bool(np.isfinite(self.triangles).all()):
            raise ValueError("triangles must be finite")

        kappa0 = self.triangles.sel(param="kappa0")
        nu0 = self.triangles.sel(param="nu0")
        ell = self.triangles.sel(param="l")

        if not bool((kappa0 >= 0).all()):
            raise ValueError("kappa0 must be >= 0")

        if not bool((ell > 0).all()):
            raise ValueError("triangle l must be > 0")

        if self.nus.ndim != 1:
            raise ValueError("nus must be 1D")

        if self.nus.sizes["gpt"] < 2:
            raise ValueError("nus must contain at least two points")

        if not bool(np.isfinite(self.nus).all()):
            raise ValueError("nus must be finite")

        if not np.all(np.diff(self.nus.values) > 0):
            raise ValueError("nus must be strictly increasing")

        if not np.isfinite(self.pref) or self.pref < 0:
            raise ValueError("pref must be finite and >= 0")
        if self.dnus.ndim != 1:
            raise ValueError("dnus must be 1D")

        if "gpt" not in self.dnus.dims:
            raise ValueError("dnus must have dimension gpt")

        if self.dnus.sizes["gpt"] != self.nus.sizes["gpt"]:
            raise ValueError("dnus must have the same length as nus")

        if not bool(np.isfinite(self.dnus).all()):
            raise ValueError("dnus must be finite")

        if not bool((self.dnus > 0).all()):
            raise ValueError("dnus must be positive")
