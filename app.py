from flask import Flask, jsonify, render_template, redirect, url_for, request
import psycopg2
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
from pykrige.ok import OrdinaryKriging
import numpy as np
import seaborn as sns
import os
from datetime import date
app = Flask(__name__)

MAPA_ZMIENNYCH = {
    "pm25": ("pm25", "pm25"),
    "temp": ("temp", "temperatura"),
    "hum": ("hum", "hum")
}

def connect_db():
    return psycopg2.connect(
        host='silesiaairpostgresql.postgres.database.azure.com',
        dbname='silesiaairdb',
        port='5432',
        user='silesiaair_admin',
        password='Superbazadanych!',
        sslmode='require'
    )

@app.route("/dane")
def dane():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM miasta;")
    dane = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(dane)

def rysuj_mape_dla_daty(data_pomiaru: str, cur, zmienna: str = "pm25"):
    tabela, kolumna = MAPA_ZMIENNYCH[zmienna]
    cur.execute(f"""
        SELECT m.nazwa, m.lat, m.lon, p.{kolumna}
        FROM miasta m
        JOIN {tabela} p ON m.id = p.miasto_id
        WHERE p.data = %s
    """, (data_pomiaru,))
    rows = cur.fetchall()
    if not rows:
        print(f"Brak danych dla daty {data_pomiaru}")
        return
    df = pd.DataFrame(rows, columns=["nazwa", "lat", "lon", zmienna])
    df["geometry"] = df.apply(lambda r: Point(r["lon"], r["lat"]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax, column=zmienna, cmap="Reds", legend=True, markersize=150, alpha=0.8)
    for _, row in gdf.iterrows():
        ax.text(row.geometry.x + 10000, row.geometry.y, row["nazwa"], fontsize=9)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.margins(0.1)
    plt.title(f"{zmienna.upper()} – {data_pomiaru}")
    plt.tight_layout()
    plt.show()

def rysuj_mape_idw(data_pomiaru: str, cur, zmienna: str = "pm25"):
    tabela, kolumna = MAPA_ZMIENNYCH[zmienna]
    cur.execute(f"""
        SELECT m.nazwa, m.lat, m.lon, p.{kolumna}
        FROM miasta m
        JOIN {tabela} p ON m.id = p.miasto_id
        WHERE p.data = %s
    """, (data_pomiaru,))
    rows = cur.fetchall()
    if not rows:
        print(f"Brak danych dla daty {data_pomiaru}")
        return
    df = pd.DataFrame(rows, columns=["nazwa", "lat", "lon", zmienna])
    df["geometry"] = df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)
    x, y, z = gdf.geometry.x.values, gdf.geometry.y.values, gdf[zmienna].values

    buffer = 20000
    grid_x = np.linspace(x.min() - buffer, x.max() + buffer, 300)
    grid_y = np.linspace(y.min() - buffer, y.max() + buffer, 300)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    def idw(xi, yi, x, y, z, power=2):
        dist = np.sqrt((x - xi) ** 2 + (y - yi) ** 2)
        dist[dist == 0] = 1e-10
        weights = 1 / dist ** power
        return np.sum(weights * z) / np.sum(weights)

    zgrid = np.zeros_like(grid_x)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            zgrid[i, j] = idw(grid_x[i, j], grid_y[i, j], x, y, z)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        zgrid,
        extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
        origin='lower',
        cmap='Reds',
        alpha=0.7
    )
    gdf.plot(ax=ax, color='black', markersize=50, edgecolor='white')
    for _, row in gdf.iterrows():
        ax.text(row.geometry.x + 2000, row.geometry.y, row["nazwa"], fontsize=8, color='black')
    plt.colorbar(im, ax=ax, label=f"Interpolowane {zmienna.upper()} (IDW)")
    ax.set_title(f"Interpolacja {zmienna.upper()} metodą IDW – {data_pomiaru}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def rysuj_mape_kriging(data_pomiaru: str, cur, zmienna: str = "pm25"):
    tabela, kolumna = MAPA_ZMIENNYCH[zmienna]
    cur.execute(f"""
        SELECT m.nazwa, m.lat, m.lon, p.{kolumna}
        FROM miasta m
        JOIN {tabela} p ON m.id = p.miasto_id
        WHERE p.data = %s
    """, (data_pomiaru,))
    rows = cur.fetchall()
    if not rows:
        print(f"Brak danych dla daty {data_pomiaru}")
        return
    df = pd.DataFrame(rows, columns=["nazwa", "lat", "lon", zmienna])
    df["geometry"] = df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)
    x, y, z = gdf.geometry.x.values, gdf.geometry.y.values, gdf[zmienna].values
    buffer = 20000
    gridx = np.linspace(x.min() - buffer, x.max() + buffer, 300)
    gridy = np.linspace(y.min() - buffer, y.max() + buffer, 300)

    ok = OrdinaryKriging(x, y, z, variogram_model="linear", verbose=False, enable_plotting=False)
    zgrid, _ = ok.execute("grid", gridx, gridy)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        zgrid.T,
        extent=(gridx.min(), gridx.max(), gridy.min(), gridy.max()),
        origin="lower",
        cmap="Reds",
        alpha=0.7
    )
    gdf.plot(ax=ax, color="black", markersize=50, edgecolor="white")
    for _, row in gdf.iterrows():
        ax.text(row.geometry.x + 2000, row.geometry.y, row["nazwa"], fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label=f"Interpolowane {zmienna.upper()}")
    ax.set_title(f"Interpolacja {zmienna.upper()} (Kriging) – {data_pomiaru}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def rysuj_wykresy_dla_daty(data_pomiaru: str, cur, zmienna: str = "pm25"):
    tabela, kolumna = MAPA_ZMIENNYCH[zmienna]
    cur.execute(f"""
        SELECT m.nazwa, p.{kolumna}
        FROM miasta m
        JOIN {tabela} p ON m.id = p.miasto_id
        WHERE p.data = %s
    """, (data_pomiaru,))
    rows = cur.fetchall()
    if not rows:
        print(f"Brak danych dla daty {data_pomiaru}")
        return
    df = pd.DataFrame(rows, columns=["nazwa", zmienna])
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Wykresy {zmienna.upper()} dla daty {data_pomiaru}", fontsize=16)

    sns.barplot(data=df, x="nazwa", y=zmienna, ax=axs[0], palette="Reds_r")
    axs[0].set_title(f"{zmienna.upper()} – wykres słupkowy")
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right', fontsize=8)

    sns.lineplot(data=df, x="nazwa", y=zmienna, marker="o", ax=axs[1], color="red")
    axs[1].set_title(f"{zmienna.upper()} – wykres liniowy")
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right', fontsize=8)

    sns.boxplot(data=df, y=zmienna, ax=axs[2], color="lightcoral")
    axs[2].set_title(f"{zmienna.upper()} – wykres pudełkowy")
    axs[2].set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generuj', methods=['POST'])
def generuj():
    data = request.form['data']
    zmienna = request.form['zmienna']
    metoda = request.form['metoda']

    conn = connect_db()
    cur = conn.cursor()

    # Ustaw ścieżkę zapisu
    output_path = os.path.join("static", "wykres.png")

    # Zmieniamy matplotlib backend na 'Agg' (do generowania bez GUI)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Wybór funkcji
    if metoda == "mapa":
        rysuj_mape_dla_daty(data, cur, zmienna)
    elif metoda == "idw":
        rysuj_mape_idw(data, cur, zmienna)
    elif metoda == "kriging":
        rysuj_mape_kriging(data, cur, zmienna)
    elif metoda == "wykres":
        rysuj_wykresy_dla_daty(data, cur, zmienna)
    else:
        return "Nieprawidłowa metoda", 400

    # Zapisuj ostatni wykres do pliku
    plt.savefig(output_path)
    plt.close()

    cur.close()
    conn.close()
    return redirect(url_for('wynik'))

@app.route('/wynik')
def wynik():
    # renderuje wykres zapisany jako statyczny plik
    return render_template('wynik.html', obraz=url_for('static', filename='wykres.png'))

if __name__ == '__main__':
    app.run(debug=True)