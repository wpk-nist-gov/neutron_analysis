from collections.abc import Iterable

import numpy as np
import pandas as pd
import periodictable

from .cached_decorators import gcached
from .utils import (
    from_index_or_col,
    keys_in_names,
    loc_levels,
    read_table_once,
    reorder_index_levels,
)


def scrape_attenuation_table(atomic_number, add_element=True):

    if atomic_number is None:
        atomic_number = range(1, 93)

    if isinstance(atomic_number, Iterable):
        add_element = True
        return pd.concat(
            (
                scrape_attenuation_table(i, add_element=add_element)
                for i in atomic_number
            )
        )

    assert 0 < atomic_number <= 92

    url = (
        "https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z{:0>2}.html".format(
            atomic_number
        )
    )

    df = (
        pd.read_html(url, skiprows=[0, 1, 3, 4], header=0)[0]
        .dropna(how="all")
        .dropna(axis=1, how="all")
    )

    names = ["tag", "energy", "mu_rho", "muen_rho"]
    if len(df.columns) == 4:
        df.columns = names
    else:
        df.columns = names[1:]
    # df.columns = ['energy', 'mu_rho','muen_rho']

    if add_element:
        df = df.assign(element=str(periodictable.elements[atomic_number]))

    return df


def fudge_atten(table, fudge=1e-6):
    """
    There are duplicate energies in attenuation table

    To get around this, and to bracket interpolation correctly,
    just fudge the duplicated values up by fudge factor
    """

    L = []
    assert fudge < 1e-5

    for element, g in table.groupby("element"):
        L.append(
            g.assign(energy=lambda x: x["energy"] + x["energy"].duplicated() * fudge)
        )

    return pd.concat(L)


def fudge_atten_twosided(table, fudge=1e-8):
    """
    There are duplicate energies in attenuation table

    To get around this, and to bracket interpolation correctly,
    just fudge the duplicated values up and down by fudge factor
    """

    table = table.reset_index(drop=True)

    L = []
    assert fudge < 1e-5

    for element, g in table.groupby("element"):

        duplicated = g["energy"].duplicated()

        if duplicated.any():

            idx_ub = duplicated[duplicated].index
            idx_lb = idx_ub - 1

            g.loc[idx_ub, "energy"] += fudge
            g.loc[idx_lb, "energy"] -= fudge

        L.append(g)

    return pd.concat(L)


def get_default_atten_table(path=None, fudge=1e-8, **kws):
    if path is None:
        from pathlib import Path

        path = Path(__file__).parent / "data" / "xray_mass_attenuation_coefs.csv"

    out = read_table_once(path, **kws)
    if fudge is not None:
        out = fudge_atten_twosided(out, fudge)

    if "tag" in out.columns:
        out = out.drop("tag", axis=1)

    return out


def interp_atten_table(elements, energies, table=None):
    if table is None:
        table = get_default_atten_table()

    idx_energies = pd.Index(energies, name="energy")
    table = table.query("element in @elements").set_index("energy")
    L = []
    for element, g in table.groupby("element"):
        L.append(
            g.reindex(g.index.union(idx_energies).sort_values())
            .interpolate("index")
            .loc[idx_energies]
            .assign(element=element)
        )

    return pd.concat(L).set_index("element", append=True)


class PhotonAtten:
    def __init__(self, gamma_table, mass_table):
        # check gamma_table:
        keys_in_names("element", gamma_table.index.names)
        keys_in_names("energy", gamma_table.columns, gamma_table.index.names)

        # check mass table:
        keys_in_names(
            ["element", "radius", "length", "density"],
            mass_table.index.names,
        )
        keys_in_names("mass_frac", mass_table.columns)

        self.gamma_table = gamma_table
        self.mass_table = mass_table

    @property
    def sample_order(self):
        return self.mass_table.index.get_level_values("sample").unique()

    @property
    def element_order(self):
        return self.gamma_table.index.get_level_values("element").unique()

    @gcached()
    def atten_table_simple(self):
        elements = from_index_or_col(self.mass_table, "element").unique()
        energy = from_index_or_col(self.gamma_table, "energy")
        return interp_atten_table(elements, energy)

    @gcached()
    def atten_table(self):
        atten_table = self.atten_table_simple
        out = (
            pd.merge(
                self.mass_table["mass_frac"],
                atten_table,
                left_index=True,
                right_index=True,
                # how='left',
            )
            .rename_axis(index={"element": "element_other"})
            .merge(
                self.gamma_table.reset_index().set_index("energy")["element"],
                left_index=True,
                right_index=True,
                how="right",
            )
            .set_index("element", append=True)
            .pipe(
                reorder_index_levels, order=[..., "element", "energy", "element_other"]
            )
            .pipe(
                loc_levels,
                indexer={"sample": self.sample_order, "element": self.element_order},
            )
        )

        return out

    @gcached()
    def atten_table_tot(self):
        df = self.atten_table

        group_cols = [x for x in df.index.names if x not in ["element_other"]]

        fac = from_index_or_col(df, "length") * from_index_or_col(df, "density")

        out = (
            df.assign(atten_coef=lambda x: fac * x["mu_rho"] * x["mass_frac"])
            .groupby(group_cols, sort=False)[["atten_coef"]]
            .sum()
            .assign(
                self_atten_coef=lambda x: (1 - np.exp(-x.atten_coef)) / x.atten_coef
            )
        )

        # order frame
        # return loc_levels(out, {'sample': self.sample_order, 'element': self.element_order})
        return out

    def atten_ratio(
        self,
        std_val="std",
        exclude_std=False,
        ratio_name=None,
        uncert_name="uncert",
        value_name=None,
        ref_levels=("element", "energy"),
    ):
        """
        ratio of "std_val" to other samples


        ratio = "std_val" / sample

        uncert = |ratio - 1| / 2 / np.sqrt(6)
        """

        # ratio
        table = self.atten_table_tot[["self_atten_coef"]]
        ref = table.query("sample==@std_val").droplevel(
            [x for x in table.index.names if x not in ref_levels]  # , 'energy']]
        )

        if exclude_std:
            table = table.query("sample != @std_val")

        ratio = ref / table

        uncert = np.abs(ratio - 1) / 2 / np.sqrt(6)

        if ratio_name is None:
            ratio_name = f"{std_val}/sample"

        d = {ratio_name: ratio, uncert_name: uncert}

        # make sure in correct order
        d = {k: reorder_index_levels(v, table.index.names) for k, v in d.items()}

        if value_name is not None:
            d = dict({value_name: table}, **d)

        return pd.concat(d, axis=1).loc[table.index]
