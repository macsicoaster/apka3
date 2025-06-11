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
from datetime import date, datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine
from azure.storage.blob import BlobServiceClient
import io

load_dotenv()

app = Flask(__name__)

MAPA_ZMIENNYCH = {
    "pm25": ("pm25", "pm25"),
    "temp": ("temp", "temperatura"),
    "hum": ("hum", "hum")
}

def connect_db():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        dbname=os.getenv('DB_NAME'),
        port=os.getenv('DB_PORT'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        sslmode=os.getenv('DB_SSLMODE', 'require')
    )

def get_db_config():
    return {
        'host': os.getenv('DB_HOST'),
        'dbname': os.getenv('DB_NAME'),
        'port': os.getenv('DB_PORT'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

def get_blob_config():
    return {
        'connection_string': os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
        'container_name': os.getenv('AZURE_STORAGE_CONTAINER_NAME')
    }

def zapisz_zjoinowana_tabele_do_blob(zmienna, blob_name=None):
    
    if zmienna not in MAPA_ZMIENNYCH:
        raise ValueError(f"Nieznana zmienna: {zmienna}")

    tabela, kolumna = MAPA_ZMIENNYCH[zmienna]

    # utworz nazwe pliku
    if blob_name is None:
        blob_name = f"{zmienna}_export_{date.today()}.csv"

    csv_filename = f"tmp_{blob_name}"

    # polaczenie z baza
    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    # pobierz zjoinowana tabele
    query = f"""
        SELECT m.nazwa, m.lat, m.lon, p.data, p.{kolumna}
        FROM miasta m
        JOIN {tabela} p ON m.id = p.miasto_id
    """

    df = pd.read_sql_query(query, engine)
    df.to_csv(csv_filename, index=False)

    # upload do blob
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
    blob_client = blob_service_client.get_blob_client(
        container=os.getenv('AZURE_STORAGE_CONTAINER_NAME'), 
        blob=blob_name
    )

    with open(csv_filename, 'rb') as data:
        blob_client.upload_blob(data, overwrite=True)

    os.remove(csv_filename)

    return blob_name

def zapisz_obraz_do_blob(plt, zmienna, data_pomiaru, metoda):
    
    # generuj unikalną nazwę pliku
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob_name = f"{zmienna}_{metoda}_{data_pomiaru}_{timestamp}.png"
    
    # zapisz obraz do bufora w pamięci
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    
    # upload do bloba
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
    blob_client = blob_service_client.get_blob_client(
        container=os.getenv('AZURE_STORAGE_CONTAINER_NAME'),
        blob=blob_name
    )
    
    blob_client.upload_blob(img_buffer, overwrite=True)
    
    return blob_name

@app.route('/eksportuj-do-blob', methods=['POST'])
def eksportuj_do_blob():
    zmienna = request.form['zmienna']
    try:
        blob_name = zapisz_zjoinowana_tabele_do_blob(zmienna)
        return jsonify({
            'status': 'success',
            'message': f'Dane zostały zapisane do Azure Blob Storage jako {blob_name}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/eksportuj-obraz-do-blob', methods=['POST'])
def eksportuj_obraz_do_blob():
    data_pomiaru = request.form['data']
    zmienna = request.form['zmienna']
    metoda = request.form['metoda']
    
    conn = connect_db()
    cur = conn.cursor()
    
    try:
        plt.figure()
        
        if metoda == "mapa":
            rysuj_mape_dla_daty(data_pomiaru, cur, zmienna)
        elif metoda == "idw":
            rysuj_mape_idw(data_pomiaru, cur, zmienna)
        elif metoda == "kriging":
            rysuj_mape_kriging(data_pomiaru, cur, zmienna)
        elif metoda == "wykres":
            rysuj_wykresy_dla_daty(data_pomiaru, cur, zmienna)
        else:
            return jsonify({
                'status': 'error',
                'message': 'Nieprawidłowa metoda wizualizacji'
            }), 400
        
        # zapis do bloba
        blob_name = zapisz_obraz_do_blob(plt, zmienna, data_pomiaru, metoda)
        plt.close()
        
        return jsonify({
            'status': 'success',
            'message': f'Obraz został zapisany do Azure Blob Storage jako {blob_name}',
            'blob_name': blob_name
        })
        
    except Exception as e:
        plt.close()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        cur.close()
        conn.close()

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generuj', methods=['POST'])
def generuj():
    data = request.form.get('data', '').strip()
    zmienna = request.form.get('zmienna', 'pm25').strip()
    metoda = request.form.get('metoda', 'mapa').strip()

    if not data:
        return "Data pomiaru jest wymagana", 400
    if zmienna not in MAPA_ZMIENNYCH:
        return "Nieprawidłowa zmienna", 400

    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()

        tabela, kolumna = MAPA_ZMIENNYCH[zmienna]
        cur.execute(f"""
            SELECT EXISTS(
                SELECT 1 FROM {tabela} 
                WHERE data = %s LIMIT 1
            )
        """, (data,))
        date_exists = cur.fetchone()[0]
        
        if not date_exists:
            return f"Brak danych dla wybranej daty: {data}", 404

        output_path = os.path.join("static", "wykres.png")

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

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

        plt.savefig(output_path)
        plt.close()

        return redirect(url_for('wynik', data=data, zmienna=zmienna, metoda=metoda))

    except Exception as e:
        app.logger.error(f"Błąd podczas generowania wykresu: {str(e)}")
        return f"Wystąpił błąd: {str(e)}", 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

@app.route('/wynik')
def wynik():
    data = request.args.get('data', '')
    zmienna = request.args.get('zmienna', 'pm25')
    metoda = request.args.get('metoda', 'mapa')
    
    return render_template(
        'wynik.html',
        obraz=url_for('static', filename='wykres.png'),
        data=data,
        zmienna=zmienna,
        metoda=metoda
    )
