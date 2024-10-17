import geopandas
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pandas as pd

pd.options.mode.chained_assignment = None
from geodatasets import get_path
import pgeocode as pgc

from datetime import datetime as dt

import plotly.express as px
import nbformat
import contextily as cx
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RectangleSelector

# create subset
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase


"""
------------------------------------------------------------------
Plotting
------------------------------------------------------------------
"""


def get_world():
    worldpath = get_path("naturalearth.land")
    world = geopandas.read_file(worldpath)
    return world


def plot_dual_map(gdf,
    bounds=[45, 28, 53, 42],
    country_codes=["UKR", "RUS"],
    colormaps=["winter", "autumn"],
    cbar_pos=[(0.1, 0.1, 0.2, 0.02), (0.75, 0.90, 0.2, 0.02)],
    labels=["Ukraine (incl. occupied territory)", "Russia"],
    title="FIRMS data map",
    normalise=True,
    data_col = "frp",
    date_col="acq_date",
    date_select = None,
):
    if date_select:
        min_date = dt.strptime(date_select[0], "%Y-%m-%d").date()
        max_date = dt.strptime(date_select[1], "%Y-%m-%d").date()
        gdf=gdf[(gdf[date_col]>min_date) & (gdf[date_col]<max_date)]

    world = get_world()
    long_min, lat_min, long_max, lat_max = bounds

    # set our extent
    w = lat_max - lat_min
    h = long_max - long_min
    ratio = w / h

    # Get dates for title
    start_date = gdf[date_col].min().strftime('%d/%m/%y')
    end_date = gdf[date_col].max().strftime('%d/%m/%y')
    title = title + f" ({start_date} - {end_date})"

    _, ax = plt.subplots(figsize=(20, 20 / ratio), layout="constrained")
    world.plot(ax=ax, alpha=0, edgecolor=[0, 0, 0, 0])
    ax.set_xlim([lat_min, lat_max])
    ax.set_ylim([long_min, long_max])
    # ax.set(title=f'"Wildfires" in Russia & Ukraine, past {past_n_days} days')
    ax.set(title=title)

    cx.add_basemap(
        ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.DarkMatterNoLabels
    )
    if normalise:
        mean_br = gdf[data_col].mean()
        std_br = gdf[data_col].std()
        gdf[f"{data_col} norm)"] = gdf[data_col].apply(
            lambda x: (x - mean_br) / std_br
        )
        plot_col = f"{data_col} norm)"
    else:
        plot_col =data_col

    for country, cmap in zip(
        country_codes,
        colormaps,
    ):
        add_points(gdf, ax=ax, cmap=cmap, country_code=country, plot_col=plot_col)
    # Add map tiles
    cx.add_basemap(
        ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.DarkMatterOnlyLabels
    )
    # cbaxes = inset_axes(ax, width="25%", height="2%", loc="lower left")
    for pos, label, cbar in zip(cbar_pos, labels, colormaps):
        add_cbar(ax=ax, cmap=cbar, cbar_pos=pos, cbar_label=label)
    cx.add_basemap(
        ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.DarkMatterOnlyLabels
    )
    return ax

def plot_gdf(
    gdf,
    bounds=[45, 28, 53, 42],
    colormap="autumn",
    cbar_pos=(0.75, 0.90, 0.2, 0.02),
    title="FIRMS data map",
    normalise=True,
    scale_by="markersize",
    data_col = "frp",
    date_col="acq_date",
    date_select = None,
):
    if date_select:
        min_date = dt.strptime(date_select[0], "%Y-%m-%d").date()
        max_date = dt.strptime(date_select[1], "%Y-%m-%d").date()
        gdf=gdf[(gdf[date_col]>min_date) & (gdf[date_col]<max_date)]

    world = get_world()
    long_min, lat_min, long_max, lat_max = bounds

    # set our extent
    w = lat_max - lat_min
    h = long_max - long_min
    ratio = w / h

    # Get dates for title
    start_date = gdf[date_col].min().strftime('%d/%m/%y')
    end_date = gdf[date_col].max().strftime('%d/%m/%y')
    title = title + f" ({start_date} - {end_date})"

    _, ax = plt.subplots(figsize=(20, 20 / ratio), layout="constrained")
    world.plot(ax=ax, alpha=0, edgecolor=[0, 0, 0, 0])
    ax.set_xlim([lat_min, lat_max])
    ax.set_ylim([long_min, long_max])
    # ax.set(title=f'"Wildfires" in Russia & Ukraine, past {past_n_days} days')
    ax.set(title=title)

    cx.add_basemap(
        ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.DarkMatterNoLabels
    )
    if normalise:
        mean_br = gdf[data_col].mean()
        std_br = gdf[data_col].std()
        gdf[f"{data_col} norm)"] = gdf[data_col].apply(
            lambda x: (x - mean_br) / std_br
        )
        plot_col = f"{data_col} norm)"
    else:
        plot_col =data_col

    gdf["markersize_"] = gdf["markersize"].apply(lambda x: ax.transData.transform([x,0])[0] - ax.transData.transform([0,0])[0])
    gdf["markersize"] = np.pi * gdf["markersize_"]**2
    
    add_points(gdf, ax=ax, cmap=colormap,  plot_col=plot_col, scale_by=scale_by)
    cx.add_basemap(
        ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.DarkMatterOnlyLabels
    )
    add_cbar(ax=ax, cmap=colormap, cbar_pos=cbar_pos, cbar_label=data_col)
    cx.add_basemap(
        ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.DarkMatterOnlyLabels
    )
    return ax


def add_points(df, ax, cmap, plot_col, scale_by="markersize"):
    df = df.sort_values(by=plot_col, ascending=True)
    df.plot(
        ax=ax,
        column=plot_col,
        zorder=1,
        cmap=cmap,
        alpha=0.8,
        markersize=scale_by,
        legend=False,
    )


def add_cbar(ax, cmap, cbar_pos, cbar_label):
    cbaxes = ax.inset_axes(cbar_pos)
    cbar = ColorbarBase(cbaxes, cmap=cmap, orientation="horizontal")
    cbar.outline.set_color("gainsboro")
    cbar.outline.set_linewidth(2)
    cbar.set_label(cbar_label, color="gainsboro")
    cbar.ax.xaxis.set_tick_params(color="gainsboro")
    plt.setp(plt.getp(cbaxes, "xticklabels"), color="gainsboro")
    cbar.outline.set_edgecolor("gainsboro")


def line_select_callback(eclick, erelease):
    pass


def make_selection(ax):
    r_select = RectangleSelector(
        ax,
        line_select_callback,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="data",
        use_data_coordinates=True,
        interactive=True,
    )
    return r_select


def select_plot(
    gdf,
    bounds=[45, 28, 53, 42],
    country_codes=["UKR", "RUS"],
    colormaps=["winter", "autumn"],
    cbar_pos=[(0.1, 0.1, 0.2, 0.02), (0.75, 0.90, 0.2, 0.02)],
    labels=["Ukraine (incl. occupied territory)", "Russia"],
    title="FIRMS data map",
):
    global selection_finished
    global waiting_for_selection
    waiting_for_selection = True
    selection_finished = False

    ax = plot_gdf(
        gdf,
        bounds=bounds,
        country_codes=country_codes,
        colormaps=colormaps,
        cbar_pos=cbar_pos,
        labels=labels,
        title=title,
    )

    r_select = make_selection(ax)
    return ax, r_select


def daily_timeseries(gdf):
    df_daily = gdf[["acq_date", "brightness"]].groupby(["acq_date"]).sum().reset_index()
    df_daily["acq_date"] = pd.to_datetime(df_daily["acq_date"]).dt.date

    fig = px.line(df_daily, x="acq_date", y="brightness")
    fig.show()
    return fig


def plot_query_timeseries(agg_df):
    df_daily = (
        agg_df[["acq_date", "location", "location_cyr", "brightness"]]
        .groupby(["acq_date", "location", "location_cyr"])
        .sum()
        .reset_index()
    )
    df_daily["acq_date"] = pd.to_datetime(df_daily["acq_date"]).dt.date
    fig = px.line(df_daily, x="acq_date", y="brightness", color="location")
    fig.show()
    return fig


"""

Dataframe manipulation

"""


def pad_times(time):
    if len(time) < 4:
        diff = 4 - len(time)
        prefix = "0" * diff
        return prefix + time
    else:
        return time


def date_fmt(gdf):
    gdf["acq_time (str)"] = gdf["acq_time"].astype(str).apply(lambda x: pad_times(x))
    gdf["datetime"] = pd.to_datetime(
        gdf["acq_date"].astype(str) + " " + gdf["acq_time (str)"].astype(str),
        format="%Y-%m-%d %H%M",
    )
    gdf["acq_date"] = pd.to_datetime(gdf["acq_date"]).dt.date
    return gdf


def select_gps_box(gdf, bbox):
    # bbox: [lat_min, long_min, lat_max, long_max]
    return gdf[
        (gdf["latitude"] > bbox[0])
        & (gdf["longitude"] > bbox[1])
        & (gdf["latitude"] < bbox[2])
        & (gdf["longitude"] < bbox[3])
    ]


def query_locations(gdf, loc_dict, nomi_code="ua", box_size=50):
    nomi = pgc.Nominatim(nomi_code)
    for idx, loc_en in enumerate(loc_dict.keys()):
        loc_cyr = loc_dict[loc_en]
        loc_df = nomi.query_location(loc_cyr, top_k=1, fuzzy_threshold=90)

        lat = loc_df["latitude"].values[0]
        long = loc_df["longitude"].values[0]

        # 1km is (very) approximately 0.01 degrees of lat/long unless near the poles
        # so for [x1, y1, x2, y2] bounding box
        box = [
            lat - box_size / 100,
            long - box_size / 100,
            lat + box_size / 100,
            long + box_size / 100,
        ]
        focus_df = select_gps_box(gdf, box)
        focus_df["location"] = loc_en
        focus_df["location_cyr"] = loc_cyr
        if idx == 0:
            agg_df = focus_df
        else:
            agg_df = pd.concat([agg_df, focus_df])
    return agg_df


"""
------------------------------------------------------------------
Data API calls
------------------------------------------------------------------
"""


def get_map_key():
    with open("data/nasa_key.txt", "r") as file:
        return file.read().rstrip()


def get_url(MAP_KEY):
    return (
        "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY="
        + MAP_KEY
    )


def get_transaction_count(URL):
    count = 0
    try:
        df = pd.read_json(URL, typ="series")
        count = df["current_transactions"]
    except:
        print("Error in our call.")
    return count


def get_data_sources(MAP_KEY):
    da_url = (
        "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/"
        + MAP_KEY
        + "/all"
    )
    return pd.read_csv(da_url)


def df_to_gdf(df, crs="EPSG:4326"):
    return geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df.longitude, df.latitude),
        crs=crs,
    )


def get_country_data(map_key = None, country_code = None, sensor=None, start_date="", n_days=10):
    # start date fmt: %Y-%m-%d
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{map_key}/{sensor}/{country_code}/{n_days}/{start_date}"
    df = pd.read_csv(url)
    df["country"] = country_code
    return df

