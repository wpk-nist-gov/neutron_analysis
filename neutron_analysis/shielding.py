import numpy as np
import pandas as pd
import periodictable

from .cached_decorators import gcached
from .utils import from_index_or_col, keys_in_names, read_table_once


def get_default_sigma_table(path=None, index_col=0, **kws):
    from pathlib import Path

    if path is None:
        path = Path(__file__).parent / "data" / "sigma_table.csv"
    return read_table_once(path, index_col=index_col, **kws)


def merge_element_props_user(table, sigma_table=None, defaults=True, **kws):
    """
    merge sigma_table (mass, sigma_a, sigma_s) onto table

    Parameters
    ----------
    table : input pandas.Series or pandas.DataFrame

    defaults : bool, default=True
        if True apply default values

        * left_on : 'element'
        * right_on : 'element
        * how: 'left'


    kws : dict
        extra keyword arguments to pandas.merge
        Note that if using `default=True`, then kws overrides these values
    """

    if defaults:
        kws = dict(dict(left_index=True, right_index=True, how="left"), **kws)

    if sigma_table is None:
        sigma_table = get_default_sigma_table()

    return pd.merge(table, sigma_table, **kws)


def merge_element_props_periodictable(table, defaults=True, **kws):
    # construct property table
    if "element" in table.index.names:
        elements = table.index.get_level_values("element").unique()

    elif "element" in table.columns:
        elements = table.elements.unique()

    else:
        raise ValueError('table must include "element" index or column')

    props = []
    for element in elements:
        e = getattr(periodictable, element)
        props.append([element, e.mass, e.neutron.absorption, e.neutron.total])

    props = pd.DataFrame(
        props, columns=["element", "atomic_mass", "sigma_a", "sigma_s"]
    ).set_index("element")

    if defaults:
        kws = dict(dict(left_index=True, right_index=True, how="left"), **kws)
    return pd.merge(table, props, **kws)


def get_Sigma(mass_frac, density, atomic_mass, sigma, use_constants=False):

    if use_constants:
        fac = periodictable.constants.avogadro_number * 1e-24
    else:
        fac = 0.6022
    return fac * mass_frac * density * sigma / atomic_mass


def calculate_props(radius, length, Sigma, Sigma_other):
    """
    calculate {x, f_slab, f_cyl, f0, f}
    """
    x = (radius * length) * Sigma / (radius + length)
    f_slab = 1 - 0.5 * x * np.log(1 / x) - 0.5 * x * (1.5 - 0.577216) - x ** 2 / 6
    f_cyl = 1 - 4 * x / 3 + x * x * np.log(2.0 / x) + 0.5 * x * x * (1.25 - 0.577216)
    f0 = (radius * f_slab + length * f_cyl) / (radius + length)
    f = f0 / (1 - (Sigma_other / Sigma) * (1 - f0))

    return {"x": x, "f_slab": f_slab, "f_cyl": f_cyl, "f0": f0, "f": f}


def read_mass_excel(path, sheet_name=0, extra_index=None, **kws):
    df = pd.read_excel(path, sheet_name=sheet_name, **kws).dropna(how="all")

    # df = df.ffill()
    if extra_index is None:
        extra_index = []

    fill_list = extra_index + ["radius", "length"]
    if "density" in df.columns:
        fill_list.append("density")

    for k in fill_list:
        df[k] = df[k].ffill()

    # strip element
    df["element"] = df["element"].str.strip()

    if "denisty" not in df.columns:
        df["density"] = df["mass"] / (np.pi * df["radius"] ** 2 * df["length"])

    if "mass_frac" not in df.columns and "mass_percent" in df.columns:
        df["mass_frac"] = df["mass_percent"] / 100

    index_cols = extra_index + ["radius", "length", "density", "element"]
    return df.set_index(index_cols)


class BaseNeutronSelfShielding:
    def __init__(self, mass_table, group_cols=None, check=True):
        """
        Parameters
        ----------
        mass_table : pd.Series or pandas.DataFrame
            if Series, then index should have levels ['element', 'density','radius','length']
            if DataFrame, should have those those levels in index

            * density : g/cm**3
            * radius : cm
            * length : cm

        """

        if isinstance(mass_table, pd.Series):
            mass_table = mass_table.to_series(name="mass_frac")

        if not isinstance(mass_table, pd.DataFrame):
            raise ValueError("must supply dataframe")

        inames = mass_table.index.names
        cnames = mass_table.columns

        keys_in_names(["element", "sample"], inames)
        keys_in_names(["density", "radius", "length"], inames)  # , cnames)
        keys_in_names(["mass_frac"], cnames)

        self.mass_table = mass_table

        if group_cols is None:
            group_cols = [
                x for x in self.mass_table.index.names if x not in ["element"]
            ]

        for k in ["radius", "length", "density"]:
            if k not in group_cols:
                group_cols.append(k)

        self.group_cols = group_cols

        if check:
            self.check_unique_elements()

    def assign(self, **kwargs):
        return type(self)(self.mass_table.assign(**kwargs))

    def set_index(self, keys, drop=True, append=False, verify_integrity=False):
        return type(self)(
            self.mass_table.set_index(
                keys,
                drop=drop,
                append=append,
                inplace=False,
                verify_integrity=verify_integrity,
            )
        )

    # @property
    # def sample_order(self):
    #     """
    #     input ordering for sample
    #     """
    #     return self.mass_table.index.get_level_values('sample').unique()

    # @property
    # def element_order(self):
    #     """
    #     input ordering for elements
    #     """
    #     return self.mass_table.index.get_level_values('element').unique()

    # @classmethod
    # def from_elements(
    #     cls,
    #     mass_table,
    #     density,
    #     radius,
    #     length,
    #     index_cols=None,
    #     mass_percent=False,
    #     **kws
    # ):
    #     """
    #     if have dict of form {element: mass_frac, ...} or series
    #     convert to dataframe
    #     """
    #     if isinstance(mass_table, dict):
    #         mass_table = pd.Series(mass_table)

    #     if isinstance(mass_table, pd.Series):
    #         mass_table = mass_table.rename_axis(index="element").to_table(
    #             name="mass_frac"
    #         )

    #         mass_table.index = mass_table.index.str.strip()

    #         mass_table = mass_table.assign(
    #             density=density,
    #             radius=radius,
    #             length=length,
    #         )

    #     if mass_percent:
    #         mass_table = mass_table.assign(mass_frac=lambda x: x["mass_frac"] / 100.0)

    #     if index_cols is None:
    #         index_cols = []
    #     index_cols = [x for x in index_cols if x not in ["density", "radius", "length"]]

    #     mass_table = mass_table.set_index(
    #         ["density", "radius", "length"] + index_cols, append=True
    #     )

    #     return cls(mass_table, group_cols=group_cols)

    @property
    def mass_frac_accounted(self):
        """mass fraction of accounted species"""
        return self.mass_table["mass_frac"].sum()

    def __repr__(self):
        return repr(self.mass_table)

    @property
    def sigma_table(self):
        raise NotImplementedError("to be implemented in subclass")

    @gcached()
    def sigma_table_tot(self):

        out = (
            self.sigma_table.groupby(self.group_cols, sort=False)[
                ["mass_frac", "Sigma_a", "Sigma_s"]
            ]
            .sum()
            .assign(Sigma_t=lambda x: x["Sigma_a"] + x["Sigma_s"])
        )

        # # order frame
        # return loc_levels(out, {'sample': self.sample_order})
        return out

    @gcached()
    def transmission_table(self):
        s = self.sigma_table_tot
        radius = from_index_or_col(s, "radius")

        t = (
            s.drop("mass_frac", axis=1)
            .apply(lambda x: np.exp(-x * radius))
            .rename(columns=lambda x: x.replace("Sigma", "transmission"))
        )

        return pd.merge(s, t, left_index=True, right_index=True)

    def _calc_output(self, Sigma, Sigma_other):
        df = self.sigma_table_tot
        kws = {k: from_index_or_col(df, k) for k in ["radius", "length"]}
        kws["Sigma"], kws["Sigma_other"] = df[Sigma], df[Sigma_other]

        out = calculate_props(**kws)
        return df.assign(**out)

    @property
    def output_scatter(self):
        return self._calc_output("Sigma_t", "Sigma_s")

    @property
    def output_noscatter(self):
        return self._calc_output("Sigma_a", "Sigma_s")

    @property
    def shielding_table(self):
        return pd.merge(
            self.output_scatter["f"].rename("f_scatter"),
            self.output_noscatter["f0"].rename("f0_noscatter"),
            left_index=True,
            right_index=True,
        )

    def check_unique_elements(self):
        for m, g in self.mass_table.groupby(self.group_cols):
            if not g.index.get_level_values("element").is_unique:
                raise ValueError("duplicate elements for sample {}".format(g))

    def shielding_ratio(
        self,
        std_val="std",
        exclude_std=False,
        ratio_name=None,
        uncert_name="uncert",
        value_name=None,
    ):
        """ratio of "std_val" to other samples


        ratio = "std_val" / sample

        uncert = |"std_val" - sample] / 2 / np.sqrt(6)

        """

        table = self.shielding_table

        ref = table.query("sample == @std_val").iloc[0]

        if exclude_std:
            table = table.query("sample != @std_val")

        ratio = ref / table

        uncert = np.abs(ref - table) / 2 / np.sqrt(6)

        if ratio_name is None:
            ratio_name = f"{std_val}/sample"

        d = {ratio_name: ratio, uncert_name: uncert}
        if value_name is not None:
            d = dict({value_name: table}, **d)

        return pd.concat(d, axis=1)


class NeutronShieldingUser(BaseNeutronSelfShielding):
    @gcached()
    def sigma_table(self):
        density = from_index_or_col(self.mass_table, "density")
        return self.mass_table.pipe(merge_element_props_user).assign(
            Sigma_a=lambda x: get_Sigma(x.mass_frac, density, x.atomic_mass, x.sigma_a),
            Sigma_s=lambda x: get_Sigma(x.mass_frac, density, x.atomic_mass, x.sigma_s),
        )


class NeutronShieldingpPT(BaseNeutronSelfShielding):
    @gcached()
    def sigma_table(self):
        density = from_index_or_col(self.mass_table, "density")
        return self.mass_table.pipe(merge_element_props_periodictable).assign(
            Sigma_a=lambda x: get_Sigma(
                x.mass_frac, density, x.atomic_mass, x.sigma_a, use_constants=True
            ),
            Sigma_s=lambda x: get_Sigma(
                x.mass_frac, density, x.atomic_mass, x.sigma_s, use_constants=True
            ),
        )
