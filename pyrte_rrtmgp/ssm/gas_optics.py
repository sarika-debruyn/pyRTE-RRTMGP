import numpy as np
from defaults import PLANCK_H, LIGHTSPEED, BOLTZMANN_K, GRAV, MOL_WEIGHTS


class GasOptics:
    
    def __init__(
        self,
        gas_names,
        triangle_params,
        nus,
        nu_min,
        nu_max,
        pref=1.0e5,
    ):
        # Store inputs
        self.gas_names = [g.lower() for g in gas_names]
        self.triangle_params = np.asarray(triangle_params, dtype=float)
        self.nus = np.asarray(nus, dtype=float)
        self.nu_min = float(nu_min)
        self.nu_max = float(nu_max)
        self.pref = float(pref)

        self.n_gases = len(self.gas_names)
        self.mol_weights = np.array(
            [MOL_WEIGHTS[g] for g in self.gas_names],
            dtype=float,
        )

        # Validate the Input
        self._validate_inputs()

        # Compute derived quantities (using function from kernel.py)
        self.band_lims = self.compute_band_limits(
            self.nus,
            self.nu_min,
            self.nu_max,
        )

        self.dnus = self.band_lims[1, :] - self.band_lims[0, :]

        self.absorption_coeffs = self.compute_absorption_coeffs(
            self.triangle_params,
            self.nus,
            self.n_gases,
        )
    
    # 
    def compute(self, atmos):
    
        play = atmos["play"].values
        plev = atmos["plev"].values
        tlay = atmos["tlay"].values
        tlev = atmos["tlev"].values
        tsfc = atmos["tsfc"].values

    
        vmr = np.stack(
            [atmos[gas].values for gas in self.gas_names],
            axis=0,
        )

        layer_mass = self.compute_layer_mass(
            vmr=vmr,
            plev=plev,
            mol_weights=self.mol_weights,
        )
        #Optical Depth
        tau = self.compute_tau(
            absorption_coeffs=self.absorption_coeffs,
            play=play,
            pref=self.pref,
            layer_mass=layer_mass,
        )

        lay_source = self.compute_planck_source(
            T=tlay,
            nus=self.nus,
            dnus=self.dnus,
        )

        lev_source = self.compute_planck_source(
            T=tlev,
            nus=self.nus,
            dnus=self.dnus,
        )

        sfc_source = self.compute_planck_source(
            T=tsfc,
            nus=self.nus,
            dnus=self.dnus,
        )

        col_dim, lay_dim = atmos["play"].dims
        _, lev_dim = atmos["plev"].dims
        gpt_dim = "gpt"

        return atmos.assign(
            tau=((col_dim, lay_dim, gpt_dim), tau),
            lay_source=((col_dim, lay_dim, gpt_dim), lay_source),
            lev_source=((col_dim, lev_dim, gpt_dim), lev_source),
            sfc_source=((col_dim, gpt_dim), sfc_source),
            nus=((gpt_dim,), self.nus),
            dnus=((gpt_dim,), self.dnus),
        )

    def _validate_inputs(self):
        if len(self.gas_names) == 0:
            raise ValueError("gas_names must not be empty")

        if len(set(self.gas_names)) != len(self.gas_names):
            raise ValueError("gas_names must be unique")

        for gas in self.gas_names:
            if gas not in MOL_WEIGHTS:
                raise ValueError(f"Unknown gas name: {gas}")

        if not np.isfinite(self.nu_min) or not np.isfinite(self.nu_max):
            raise ValueError("nu_min and nu_max must be finite")

        if self.nu_min >= self.nu_max:
            raise ValueError("nu_min must be less than nu_max")

        if self.nus.ndim != 1:
            raise ValueError("nus must be 1D")

        if len(self.nus) < 2:
            raise ValueError("nus must contain at least two points")

        if not np.all(np.isfinite(self.nus)):
            raise ValueError("nus must be finite")

        if not np.all(np.diff(self.nus) > 0):
            raise ValueError("nus must be strictly increasing")

        if not np.all((self.nus > self.nu_min) & (self.nus < self.nu_max)):
            raise ValueError("all nus must satisfy nu_min < nus < nu_max")

        tri = self.triangle_params

        if tri.ndim != 2 or tri.shape[1] != 4:
            raise ValueError("triangle_params must have shape (ntriangles, 4)")

        if not np.all(np.isfinite(tri)):
            raise ValueError("triangle_params must be finite")

        gas_idx = tri[:, 0]
        kappa0 = tri[:, 1]
        nu0 = tri[:, 2]
        ell = tri[:, 3]

        if not np.all(gas_idx == np.floor(gas_idx)):
            raise ValueError("triangle gas indices must be integers")

        if not np.all((gas_idx >= 1) & (gas_idx <= self.n_gases)):
            raise ValueError("triangle gas indices must lie between 1 and n_gases")

        if not np.all(kappa0 >= 0):
            raise ValueError("kappa0 must be >= 0")

        if not np.all((nu0 >= self.nu_min) & (nu0 <= self.nu_max)):
            raise ValueError("triangle nu0 must satisfy nu_min < nu0 < nu_max")

        if not np.all(ell > 0):
            raise ValueError("triangle l must be > 0")

        if not np.isfinite(self.pref) or self.pref < 0:
            raise ValueError("pref must be finite and >= 0")

    @staticmethod
    def compute_band_limits(nus, nu_min, nu_max):
        n_nu = len(nus)
        band_lims = np.empty((2, n_nu), dtype=float)

        band_lims[0, 0] = nu_min
        band_lims[0, 1:] = 0.5 * (nus[:-1] + nus[1:])

        band_lims[1, :-1] = 0.5 * (nus[:-1] + nus[1:])
        band_lims[1, -1] = nu_max

        return band_lims

    @staticmethod
    def compute_absorption_coeffs(triangle_params, nus, n_gases):
        n_nu = len(nus)
        absorption_coeffs = np.zeros((n_gases, n_nu), dtype=float)

        for row in triangle_params:
            gas_idx = int(row[0]) - 1
            kappa0 = row[1]
            nu0 = row[2]
            ell = row[3]

            absorption_coeffs[gas_idx, :] += (
                kappa0 * np.exp(-np.abs(nus - nu0) / ell)
            )

        return absorption_coeffs
    
    @staticmethod
    def compute_layer_mass(vmr: np.ndarray, plev: np.ndarray, mol_weights: np.ndarray, m_dry: float = 0.029) -> np.ndarray:
        # pressure thickness of each layer
            dp = np.abs(plev[:, 1:] - plev[:, :-1]) #always positive
        # mol_weights has shape (n_gases,); broadcast to (n_gases, n_col, n_lay)
        # dp has shape (n_col, n_lay); broadcast by adding a leading axis
            mmr = vmr * (mol_weights[:, np.newaxis, np.newaxis] / m_dry)
            layer_mass = mmr * dp[np.newaxis, :, :] / GRAV
            return layer_mass
    
    @staticmethod
    def compute_tau(absorption_coeffs: np.ndarray, play: np.ndarray, pref: float, layer_mass: np.ndarray) -> np.ndarray:

        #pressure broadening factor: shape (n_col, n_lay)
        if pref != 0.0:
            p_scaling = play / pref
        else:
            p_scaling = np.ones_like(play)

        # sum over gases: layer_mass (n_gases, n_col, n_lay), absorption_coeffs (n_gases, n_nu)
        # we want result (n_col, n_lay, n_nu)
        # for each gas g: contribution(col, lay, nu) = layer_mass(g, col, lay) * kappa(g, nu)
        # use Einstein summation: 'gcl, gn -> cln'
        gas_weighted = np.einsum('gcl,gn->cln', layer_mass, absorption_coeffs)

        # apply pressure broadening: p_scaling is (n_col, n_lay), result is (l, n_lay, n_nu)
        tau = p_scaling[:, :, np.newaxis] * gas_weighted

        return tau
    
    @staticmethod
    def compute_band_limits(nus: np.ndarray, nu_min: float, nu_max: float) -> np.ndarray:
        n_nu = len(nus) #default case is 42 wavenumber points
        band_lims = np.empty((2, n_nu), dtype=float) #empty 2D array, two rows and one column per wavenumber point

        # lower edges: nu_min for the first band, midpoints for the rest
        band_lims[0, 0] = nu_min
        band_lims[0, 1:] = 0.5 * (nus[:-1] + nus[1:])

        # upper edges: midpoints for all but last, nu_max for last
        band_lims[1, :-1] = 0.5 * (nus[:-1] + nus[1:])
        band_lims[1, -1] = nu_max

        return band_lims
    
    @staticmethod
    def planck_function(T: np.ndarray, nu: float) -> np.ndarray:
        nu_si = nu/100.0 #convert cm^-1 to m^-1
        numerator = 100.0 * 2.0 * PLANCK_H * (nu_si ** 3) * (LIGHTSPEED ** 2)
  
        exponent = (PLANCK_H * LIGHTSPEED * nu_si) / (BOLTZMANN_K * T)
        denominator = np.exp(exponent) - 1.0 
  
        return numerator / denominator
    
    @staticmethod
    def compute_planck_source(T: np.ndarray, nus: np.ndarray, dnus: np.ndarray) -> np.ndarray:
    # T has shape (...), we need to evalute B_nu at each wavenumber
    # Add a trailing axist to T so it broadcasts against nus
        T_expanded = T[..., np.newaxis] #(..., 1)

        #nus and dnus have shape (n_nu), broadcast against T_expanded -> (..., n_nu)
        source = GasOptics.planck_function(T_expanded, nus) * dnus

        return source
