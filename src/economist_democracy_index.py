import numpy as np
import pandas as pd
import wikipedia as wp
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class Colors:
    BLACK = "#000000"
    GRAY = "#7f7f7f"
    LIGHT_GRAY = "gainsboro"
    DARK_GRAY = "#404040"
    BLUE = "#1f77b4"
    ORANGE = "#ff7f0e"
    GREEN = "#2ca02c"
    RED = "#d62728"
    PURPLE = "#9467bd"
    BROWN = "#8c564b"
    PINK = "#e377c2"
    LIGHT_BLUE = "#d2e3f0"
    LIGHT_ORANGE = "#ffe5ce"
    LIGHT_GREEN = "#d4ecd4"
    LIGHT_RED = "#f6d3d4"
    LIGHT_PURPLE = "#e9e0f1"
    LIGHT_BROWN = "#e8dddb"
    LIGHT_PINK = "#f9e3f2"

    def __init__(self):

        # A custom colormap to use with Matplotlib
        self.colormaps = {
            "RdWtGr": mcolors.LinearSegmentedColormap.from_list(
                        "RdWtGr",
                        [self.RED, "white", self.GREEN],
                        N=8)
        }

        # A custom colorscale to use with Plotly
        colorscale = []
        for i in range(8):
            j = i / 8
            c = mcolors.rgb2hex(self.colormaps["RdWtGr"]((j + j + 1/8) / 2))
            colorscale.append((j, c))
            colorscale.append((j + 1 / 8, c))
        self.colorscales = {
            "RdWtGr": colorscale,
        }

    @staticmethod
    def get_opaque_hex_from_transparency(hex: str, transparency: float) -> str:
        """
        This method take a hex color and a transparency value (between 0 and 1)
        and returns the equivalent opaque color.

        Parameters
        ----------
        hex : str
            The hexadecimal color.
        transparency : float
            The transparency value. Must be between 0 and 1, where 0 is fully
            transparent and 1 is fully opaque.

        Returns
        -------
        str
            The opaque color in hexadecimal format.
        """
        hex = hex.lstrip("#")
        r, g, b = tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))
        r = int(255 - transparency * (255 - r))
        g = int(255 - transparency * (255 - g))
        b = int(255 - transparency * (255 - b))
        return "#{:02x}{:02x}{:02x}".format(r, g, b)


class Config:
    region_colors = {
        "North America": Colors.PURPLE,
        "Western Europe": Colors.PINK,
        "Eastern Europe and Central Asia": Colors.ORANGE,
        "Latin America and the Caribbean": Colors.GREEN,
        "Asia and Australasia": Colors.BLUE,
        "Middle East and North Africa": Colors.RED,
        "Sub-Saharan Africa": Colors.BROWN,
    }

    region_bg_colors = {
        "North America": Colors.LIGHT_PURPLE,
        "Western Europe": Colors.LIGHT_PINK,
        "Eastern Europe and Central Asia": Colors.LIGHT_ORANGE,
        "Latin America and the Caribbean": Colors.LIGHT_GREEN,
        "Asia and Australasia": Colors.LIGHT_BLUE,
        "Middle East and North Africa": Colors.LIGHT_RED,
        "Sub-Saharan Africa": Colors.LIGHT_BROWN,
    }


class Data:
    def __init__(self):
        self._setup_data()

    def _setup_data(self) -> None:
        """
        Load the data from the CSV file and preprocess it.
        """
        self.df = pd.read_csv(
            "data/economist-democracy-index/raw/democracy_index.csv")
        self.df = self.df[self.df.columns.drop(
            list(self.df.filter(regex=' rank')))]
        self.df["Region"] = self.df["Region"].astype("category")
        self.df["RegimeType"] = self.df["RegimeType"].astype("category")

        self.df = self.df.melt(
            id_vars=["Region", "Country", "RegimeType"],
            var_name="Year",
            value_name="DemocracyIndex"
        )
        self.df["Year"] = self.df["Year"].astype(int)

    def filter_by_region(self, regions: list[str]) -> pd.DataFrame:
        """
        Returns a DataFrame filtered by the given regions, specified as a list
        of strings.

        Parameters
        ----------
        regions : list[str]
            A list of regions to filter by.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame.
        """
        return self._filter("Region", regions)

    def filter_by_country(self, countries: list[str]) -> pd.DataFrame:
        """
        Returns a DataFrame filtered by the given countries, specified as a
        list of strings.

        Parameters
        ----------
        countries : list[str]
            A list of countries to filter by.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame.
        """
        return self._filter("Country", countries)

    def filter_by_regime(self, regimes: list[str]) -> pd.DataFrame:
        """
        Returns a DataFrame filtered by the given regime types,
        specified as a list of strings.

        Parameters
        ----------
        regimes : list[str]
            A list of regime types to filter by.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame.
        """
        return self._filter("RegimeType", regimes)

    def filter_by_year(self, year: int) -> pd.DataFrame:
        """
        Returns a DataFrame filtered by the given year, specified as an int.

        Parameters
        ----------
        year : int
            The year to filter by.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame.
        """
        return self.df[self.df["Year"] == year]

    def _filter(self, key: str, values: list[str]) -> pd.DataFrame:
        """
        Returns a DataFrame filtered by the given key and values.

        Parameters
        ----------
        key : str
            The key to filter by.
        values : list[str]
            A list of values to filter by.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame.
        """
        return self.df[self.df[key].isin(values)]

    def get_world_average(self) -> pd.DataFrame:
        """
        Returns the world average of the democracy index for each year.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the world average democracy index for each
            year.
        """
        return self.df.groupby("Year")[
            "DemocracyIndex"].mean().reset_index(name="DemocracyIndex")

    def get_region_averages(self) -> pd.DataFrame:
        """
        Returns the average democracy index for each region and year.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the average democracy index for each region
            and year.
        """
        return self.df.groupby(
            ["Region", "Year"], observed=True)[
                "DemocracyIndex"].mean().reset_index(
                    name="DemocracyIndex")

    @staticmethod
    def get_merged_dataframe() -> pd.DataFrame:
        """
        Returns a merged DataFrame of the democracy index data and the world
        countries shapefile.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the merged data.
        """
        data = Data().df
        countries = gpd.read_file(
            "data/economist-democracy-index/external/ne_110m_admin_0_"
            "countries/ne_110m_admin_0_countries.shp")

        # Use names in `data`
        to_replace = [
            "Bosnia and Herz.", "Côte d'Ivoire", "United States of America",
            "Central African Rep.", "Eq. Guinea", "Congo", "eSwatini",
            "Czechia", "Dominican Rep.", "Dem. Rep. Congo", "Timor-Leste",
            "Greenland", "Falkland Is."]
        value = [
            "Bosnia and Herzegovina", "Ivory Coast", "United States",
            "Central African Republic", "Equatorial Guinea",
            "Republic of the Congo", "Eswatini", "Czech Republic",
            "Dominican Republic", "Democratic Republic of the Congo",
            "East Timor", "Denmark", "Argentina"]
        countries.replace(to_replace=to_replace, value=value, inplace=True)

        merged_df = countries.merge(data, left_on="NAME", right_on="Country",
                                    how="left")

        return merged_df

    @staticmethod
    def get_yearly_data(year: int) -> pd.DataFrame:
        """
        Returns a DataFrame filtered by the given year, specified as an int.
        #TODO: Remove this function and use the equivalent in the Data class
        # instead.

        Parameters
        ----------
        year : int
            The year to filter by.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame.
        """
        df = Data().df
        df = df[df["Year"] == year]

        # Remap regime types to this year
        df["RegimeType"] = df["DemocracyIndex"].apply(Data.assign_regime_type)

        return df

    @staticmethod
    def get_yearly_geographic_data(year: int) -> pd.DataFrame:
        """
        Returns a DataFrame filtered by the given year, specified as an int,
        that also contains geographic data for each country.

        Parameters
        ----------
        year : int
            The year to filter by.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame.
        """
        df = Data.get_merged_dataframe()
        df = df[df["NAME"] != "Antarctica"]
        df = df[df["Year"] == year]

        # Remap regime types to this year
        df["RegimeType"] = df["DemocracyIndex"].apply(Data.assign_regime_type)

        return df

    @staticmethod
    def assign_regime_type(democracy_index: float) -> str:
        """
        Assigns a regime type based on the democracy index.

        Parameters
        ----------
        democracy_index : float
            The democracy index.

        Returns
        -------
        str
            The regime type.
        """
        regime_type = "Authoritarian"
        if democracy_index >= 8.0:
            regime_type = "Full democracy"
        elif democracy_index >= 6.0:
            regime_type = "Flawed democracy"
        elif democracy_index >= 4.0:
            regime_type = "Hybrid regime"
        return regime_type

    @staticmethod
    def get_index_change_geographic_data(start_year: int,
                                         end_year: int) -> pd.DataFrame:
        """
        Returns a DataFrame containing the democracy index change between two
        years, specified as ints, together with geographic data for each
        country.

        Parameters
        ----------
        start_year : int
            The starting year.
        end_year : int
            The ending year.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the democracy index change and geographic
            data.
        """
        df = Data.get_merged_dataframe()
        df = df[df["NAME"] != "Antarctica"]
        index_change = df[df["Year"] == end_year][
            "DemocracyIndex"].to_numpy() \
            - df[df["Year"] == start_year]["DemocracyIndex"].to_numpy()
        df = df[df["Year"] == end_year]
        df["IndexChange"] = index_change
        return df

    @staticmethod
    def get_migration_matrix(start_year: int, end_year: int) -> np.ndarray:
        """
        Calculates the regime types migration matrix (changes in regime types)
         between two years, specified as ints.

        Parameters
        ----------
        start_year : int
            The starting year.
        end_year : int
            The ending year.

        Returns
        -------
        np.ndarray
            The migration matrix.
        """
        df1 = Data.get_yearly_data(start_year)
        df2 = Data.get_yearly_data(end_year)

        regimes = ["Authoritarian", "Hybrid regime",
                   "Flawed democracy", "Full democracy"]
        n_regimes = len(regimes)

        m = np.zeros((n_regimes + 1, n_regimes + 1))
        for i, r1 in enumerate(regimes):
            for j, r2 in enumerate(regimes):
                m[i, j] = ((df1["RegimeType"].to_numpy() == r1)
                           & (df2["RegimeType"].to_numpy() == r2)).sum()
        m[-1, -1] = np.nan
        m[-1, :n_regimes] = np.sum(m[:n_regimes, :n_regimes], axis=0)
        m[:n_regimes, -1] = np.sum(m[:n_regimes, :n_regimes], axis=1)
        return m

    @staticmethod
    def get_raw_data() -> None:
        """
        Fetches the Democracy Index data from Wikipedia and saves
         it as a CSV file.
        """
        html = wp.page("The_Economist_Democracy_Index").html().encode("utf-8")

        try:
            df = pd.read_html(html)[5]
        except IndexError:
            raise ValueError(
                "The expected table was not found on the Wikipedia.")

        df.rename(columns={"Regime type": "RegimeType"}, inplace=True)
        df = df.map(
            lambda x: x.replace("Asia and Austral\xadasia",
                                "Asia and Australasia").replace(
                                    "Latin America and the Carib\xadbean",
                                    "Latin America and the Caribbean"
                                ) if isinstance(x, str) else x)
        df.to_csv("data/raw/democracy_index.csv", index=False)


def plot_evolution_regions() -> None:
    """
    Plots the evolution of the Democracy Index by region from 2006 to 2024.
    """
    data = Data()

    regions = list(data.df["Region"].unique())
    region_df = data.get_region_averages()
    y_text_position = {
        'Asia and Australasia': 5,
        'Eastern Europe and Central Asia': 5.5,
        'Latin America and the Caribbean': 6,
        'Middle East and North Africa': region_df[
            region_df["Region"] == "Middle East and North Africa"][
                "DemocracyIndex"].to_numpy()[-1],
        'Sub-Saharan Africa': region_df[
            region_df["Region"] == "Sub-Saharan Africa"][
                "DemocracyIndex"].to_numpy()[-1],
        'North America': 8,
        'Western Europe': 9,
        }

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.tight_layout(rect=(0, 0.04, 0.66, 0.84))

    for region in regions:
        region_data = region_df[region_df["Region"] == region]
        ax.plot(region_data["Year"], region_data["DemocracyIndex"],
                marker="o", label=region, color=Config.region_colors[region],
                linewidth=2, markersize=5)
        x_end = region_data["Year"].to_numpy()[-1]
        y_end = region_data["DemocracyIndex"].to_numpy()[-1]
        ax.annotate(text=region, xy=(x_end, y_end),
                    xytext=(2024.5, y_text_position[region]),
                    ha='left', va='center', fontsize=12, fontweight='bold',
                    color=Config.region_colors[region],
                    arrowprops=dict(
                        arrowstyle='-',
                        color=Config.region_colors[region],
                        lw=0.8, shrinkA=0, shrinkB=0,
                        connectionstyle="angle,angleA=0,angleB=90"
                    ),
                    annotation_clip=False)

    ax.set_ylim(0, 10.1)
    ax.set_yticks(range(0, 11))
    ax.tick_params(axis="y", labelsize=12, colors=Colors.DARK_GRAY, length=0)

    ax.set_xlim(2005.8, 2024.2)
    ax.set_xticks([year for year in range(2007, 2024, 2)])
    ax.tick_params(axis="x", labelsize=12, colors=Colors.DARK_GRAY)

    ax.grid(True, which="major", axis="y", color=Colors.LIGHT_GRAY,
            linewidth=1)
    ax.axhline(0, color=Colors.DARK_GRAY, linewidth=2)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.text(2006, 9.75, "MORE DEMOCRATIC", color=Colors.DARK_GRAY,
            fontsize=10, va="center", ha="left", fontweight="bold")
    ax.text(2006, 0.25, "LESS DEMOCRATIC", color=Colors.DARK_GRAY,
            fontsize=10, va="center", ha="left", fontweight="bold")
    ax.text(
        2005.1, 12.5, "The Economist Democracy Index, 2006 - 2024",
        ha="left", va="top",
        fontsize=20, color=Colors.DARK_GRAY, fontweight="bold")
    plt.text(
        2005.1, 11.7,
        "This chart shows the evolution of The Economist Democracy Index\n"
        "between 2006 and 2024, averaged by region.",
        ha="left", va="top", fontsize=14, color=Colors.DARK_GRAY)
    ax.text(
        2005.1, -1,
        "Source(s): The Economist/Wikipedia "
        "(https://en.wikipedia.org/wiki/The_Economist_Democracy_Index)",
        ha="left", va="top", fontsize=11, color=Colors.DARK_GRAY)

    plt.savefig("reports/economist-democracy-index/time_series_by_region.png",
                dpi=500)
    plt.close(fig)


def plot_evolution_countries() -> None:
    data = Data()
    countries = [
        ("Argentina", Colors.BLUE),
        ("Mali", Colors.ORANGE),
        ("Bhutan", Colors.GREEN),
        ("Afghanistan", Colors.RED),
        ("Norway", Colors.PURPLE),
        ("Nicaragua", Colors.BROWN),
    ]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.tight_layout(rect=(0, 0.04, 0.66, 0.84))

    for country, color in countries:
        country_data = data.df[data.df["Country"] == country]
        country_data = country_data.sort_values(by="Year")
        ax.plot(
            country_data["Year"], country_data["DemocracyIndex"],
            marker="o", label=country, color=color, linewidth=2, markersize=5
        )
        x_end = country_data["Year"].to_numpy()[-1]
        y_end = country_data["DemocracyIndex"].to_numpy()[-1]
        ax.annotate(
            text=country, xy=(x_end, y_end), xytext=(2024.5, y_end),
            ha='left', va='center', fontsize=12, fontweight='bold',
            color=color,
            arrowprops=dict(
                arrowstyle='-', color=color, lw=0.8, shrinkA=0, shrinkB=0,
                connectionstyle="angle,angleA=0,angleB=90"
            ), annotation_clip=False)

    ax.set_ylim(0, 10.1)
    ax.set_yticks(range(0, 11))
    ax.tick_params(axis="y", labelsize=12, colors=Colors.DARK_GRAY, length=0)

    ax.set_xlim(2005.8, 2024.2)
    ax.set_xticks([year for year in range(2007, 2024, 2)])
    ax.tick_params(axis="x", labelsize=12, colors=Colors.DARK_GRAY)

    ax.grid(True, which="major", axis="y", color=Colors.LIGHT_GRAY,
            linewidth=1)
    ax.axhline(0, color=Colors.DARK_GRAY, linewidth=2)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.text(2006, 9.75, "MORE DEMOCRATIC", color=Colors.DARK_GRAY,
            fontsize=10, va="center", ha="left", fontweight="bold")
    ax.text(2006, 0.25, "LESS DEMOCRATIC", color=Colors.DARK_GRAY,
            fontsize=10, va="center", ha="left", fontweight="bold")
    ax.text(
        2005.1, 12.5, "The Economist Democracy Index, 2006 - 2024",
        ha="left", va="top",
        fontsize=20, color=Colors.DARK_GRAY, fontweight="bold")
    ax.text(
        2005.1, 11.7,
        "This chart shows the evolution of The Economist Democracy Index\n"
        "between 2006 and 2024 for selected countries.",
        ha="left", va="top", fontsize=14, color=Colors.DARK_GRAY)
    ax.text(
        2005.1, -1,
        "Source(s): The Economist/Wikipedia "
        "(https://en.wikipedia.org/wiki/The_Economist_Democracy_Index)",
        ha="left", va="top", fontsize=11, color=Colors.DARK_GRAY)

    plt.savefig("reports/economist-democracy-index/time_series_by_country.png",
                dpi=500)
    plt.close(fig)


def plot_world_map_index(year: int) -> None:
    """
    Plots a world map of the Democracy Index for a given year.

    Parameters
    ----------
    year : int
        The year for which to plot the map.
    """
    df = Data.get_yearly_geographic_data(year=year)
    colors = Colors()
    cmap = mpl.colormaps.get_cmap("viridis")
    vmin, vmax = 0, 10

    fig, ax = plt.subplots(figsize=(9, 6.5))

    df.plot(
        column="DemocracyIndex", cmap=cmap, linewidth=0.5, edgecolor="white",
        ax=ax, legend=False,
        legend_kwds={
            "label": "Democracy Index",
            "orientation": "horizontal",
            "shrink": 0.7,
            "pad": 0.02,
            "aspect": 40,
            "fraction": 0.04,
            "anchor": (0.5, 0),
            "extend": "both",
            "ticks": [0, 2, 4, 6, 8, 10],
        },
        vmin=vmin, vmax=vmax,
        missing_kwds={
            "color": colors.LIGHT_GRAY,
            "edgecolor": "white",
            "hatch": "///",
            "label": "No data",
        },
    )

    ax.text(
        0, 1.05, f"The Economist Democracy Index Map, {year}",
        ha="left", va="bottom", transform=ax.transAxes,
        fontsize=18, color=Colors.DARK_GRAY, fontweight="bold")
    ax.text(
        0, 0.99,
        "This chart shows a world map of the Economist Democracy Index"
        f" in {year}.", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=13, color=Colors.DARK_GRAY)
    ax.text(
        0, -0.15,
        "Source(s): The Economist/Wikipedia "
        "(https://en.wikipedia.org/wiki/The_Economist_Democracy_Index)",
        ha="left", va="top", fontsize=10, color=Colors.DARK_GRAY,
        transform=ax.transAxes)

    ax.axis("off")

    cax = inset_axes(ax, width="60%", height="3%",
                     loc='lower center', borderpad=-2)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal',
                        ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(which='both', length=0)
    cbar.ax.tick_params(axis='x', pad=2)
    cbar.ax.xaxis.grid(True, which='both', color='white',
                       linestyle='-', linewidth=0.5)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    plt.savefig(f"reports/economist-democracy-index/map_index_{year}.png",
                dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_world_map_index_change(start_year: int, end_year: int) -> None:
    """
    Plots a world map of the change in the Democracy Index between two years.

    Parameters
    ----------
    start_year : int
        The starting year for the change calculation.
    end_year : int
        The ending year for the change calculation.
    """
    colors = Colors()
    df = Data.get_index_change_geographic_data(start_year, end_year)
    cmap = colors.colormaps["RdWtGr"]
    vmin, vmax = -4, 4

    fig, ax = plt.subplots(figsize=(9, 6.5))

    df.plot(
        column="IndexChange", cmap=cmap, linewidth=0.5, edgecolor="white",
        ax=ax, legend=False,
        vmin=vmin, vmax=vmax,
        missing_kwds={
            "color": colors.LIGHT_GRAY,
            "edgecolor": "white",
            "hatch": "///",
            "label": "No data",
        },
    )

    ax.axis("off")

    cax = inset_axes(ax, width="60%", height="3%",
                     loc='lower center', borderpad=-2)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal',
                        ticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(which='both', length=0)
    cbar.ax.tick_params(axis='x', pad=2)
    cbar.ax.xaxis.grid(True, which='both', color='white',
                       linestyle='-', linewidth=0.5)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0, 1.05,
        "The Economist Democracy Index Variation Map",
        ha="left", va="bottom", transform=ax.transAxes,
        fontsize=18, color=Colors.DARK_GRAY, fontweight="bold")
    ax.text(
        0, 0.99,
        f"Change in the Economist Democracy Index, {start_year}-{end_year}.",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=13, color=Colors.DARK_GRAY)
    ax.text(
        0, -0.15,
        "Source(s): The Economist/Wikipedia "
        "(https://en.wikipedia.org/wiki/The_Economist_Democracy_Index)",
        ha="left", va="top", fontsize=10, color=Colors.DARK_GRAY,
        transform=ax.transAxes)

    fig.tight_layout()
    plt.savefig("reports/economist-democracy-index/"
                f"map_index_change_{start_year}_to_{end_year}.png",
                dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_regions() -> None:
    df = Data.get_yearly_geographic_data(year=2006)
    config = Config()

    # Assign a color to each region
    df["RegionColor"] = df["Region"].map(config.region_colors)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for region, color in config.region_colors.items():
        region_df = df[df["Region"] == region]
        region_df.plot(ax=ax, color=color, linewidth=0.5,
                       edgecolor="white", label=region)

    ax.axis("off")

    handles = [patches.Patch(color=color, label=region)
               for region, color in config.region_colors.items()]
    ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.18, 0),
        frameon=False, fontsize=8, title_fontsize=13)

    ax.text(
        0, 1.05, "World Regions",
        ha="left", va="bottom", transform=ax.transAxes,
        fontsize=18, color=Colors.DARK_GRAY, fontweight="bold")
    ax.text(
        0, 0.99,
        "This chart shows a world map of the different regions.",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=13, color=Colors.DARK_GRAY)
    ax.text(
        0, -0.15,
        "Source(s): The Economist/Wikipedia "
        "(https://en.wikipedia.org/wiki/The_Economist_Democracy_Index)",
        ha="left", va="top", fontsize=10, color=Colors.DARK_GRAY,
        transform=ax.transAxes)

    fig.tight_layout()
    plt.savefig("reports/economist-democracy-index/map_regions.png",
                dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_regime_migration(start_year: int, end_year: int) -> None:
    """
    Plots a heatmap of regime type changes between two years.

    Parameters
    ----------
    start_year : int
        The starting year for the reigme change calculation.
    end_year : int
        The ending year for the reigme change calculation.
    """
    m = Data.get_migration_matrix(start_year, end_year)
    column_labels = ["Full\nDemocracies", "Flawed\nDemocracies",
                     "Hybrid\nRegimes", "Authoritarian\nRegimes"]

    fig, ax = plt.subplots(figsize=(8, 8))
    # fig.tight_layout(rect=(0.2, 0.2, 0.8, 0.8))

    ax.set_xlim(0, 5)
    ax.set_ylim(5, 0)

    mask = np.isnan(m)

    greens = mpl.colormaps.get_cmap("Greens")
    reds = mpl.colormaps.get_cmap("Reds")
    cmat = [
        ["white",  greens(0.25), greens(0.45), greens(0.65), "gainsboro"],
        [reds(0.25), "white", greens(0.25), greens(0.45), "gainsboro"],
        [reds(0.45), reds(0.25), "white", greens(0.25), "gainsboro"],
        [reds(0.65), reds(0.45), reds(0.25), "white", "gainsboro"],
        ["gainsboro", "gainsboro", "gainsboro", "gainsboro", "white"]
    ]
    cmat_hex = [[mcolors.to_hex(c) if isinstance(c, str) or isinstance(c, tuple) else c for c in row] for row in cmat]

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not mask[i, j]:
                color = cmat_hex[i][j]
                rect = plt.Rectangle([j, i], 1, 1, facecolor=color, edgecolor="white", linewidth=3)
                ax.add_patch(rect)
                value = str(int(m[i, j]))
                ax.text(j + 0.5, i + 0.5, value, ha="center", va="center",
                        fontsize=18, color=Colors.DARK_GRAY, weight="bold")

    # Set axis labels
    for i, label in enumerate(column_labels[::-1]):
        ax.text(i + 0.5, -0.1, label, ha="left", va="center",
                fontsize=12, color=Colors.DARK_GRAY, rotation=90,
                rotation_mode="anchor")
        ax.text(-0.1, i + 0.5, label, ha="right", va="center",
                fontsize=12, color=Colors.DARK_GRAY, rotation=0,
                rotation_mode="anchor")
    ax.text(2, -1.5 - 0.05, str(end_year), ha="center", va="bottom",
            fontsize=12, color=Colors.DARK_GRAY, weight="bold")
    ax.text(-1.5 - 0.05, 2, str(start_year), ha="center", va="bottom",
            fontsize=12, color=Colors.DARK_GRAY, weight="bold", rotation=90,
            rotation_mode="anchor")
    ax.annotate(
        '', xy=(-1.5, 0), xytext=(-1.5, 4),
        arrowprops=dict(
            arrowstyle='-',
            lw=1,
            color=Colors.DARK_GRAY,
        ), annotation_clip=False,
    )
    ax.annotate(
        '', xy=(0, -1.5), xytext=(4, -1.5),
        arrowprops=dict(
            arrowstyle='-',
            lw=1,
            color=Colors.DARK_GRAY,
        ), annotation_clip=False,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect('equal')
    ax.text(
        -0.5, 1.45, f"Changes in Regime Types, {start_year} - {end_year}",
        ha="left", va="bottom", transform=ax.transAxes,
        fontsize=18, color=Colors.DARK_GRAY, fontweight="bold")
    ax.text(
        -0.5, 1.4,
        f"This chart shows the change in regime types between {start_year} "
        f"and {end_year}.",
        fontsize=14, color=Colors.DARK_GRAY, transform=ax.transAxes, ha="left",
        va="bottom"
    )
    # ax.text(
    #     0, -0.12,
    #     "Source(s): The Economist/Wikipedia (https://en.wikipedia.org/wiki/The_Economist_Democracy_Index)",
    #     fontsize=11, color=Colors.DARK_GRAY, transform=ax.transAxes, ha="left"
    # )

    # Remove axis spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    plt.savefig("reports/economist-democracy-index/regime_migration.png",
                dpi=500, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # plot_evolution_regions()
    # plot_evolution_countries()
    # plot_world_map_index(year=2006)
    # plot_world_map_index(year=2024)
    # plot_world_map_index_change(start_year=2006, end_year=2024)
    # plot_regions()
    plot_regime_migration(start_year=2006, end_year=2024)
