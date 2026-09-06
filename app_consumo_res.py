import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib.patches import FancyBboxPatch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
)

st.set_page_config(page_title="ML: Consumo Total vs Hora (InfluxDB)", layout="wide")

def dibujar_arbol_simple(clf, feature_names, class_names, ax, colores=None):
    """Dibuja el árbol mostrando solo la condición de corte (nodos internos) y la clase (hojas),
    sin gini/samples/value — a diferencia de sklearn.tree.plot_tree.

    Además, colapsa visualmente cualquier subárbol cuyas hojas prediquen todas la misma clase:
    CART puede seguir dividiendo mientras gane pureza (Gini), aunque esa división no cambie
    la clase predicha a ningún lado — esos cortes no aportan a la decisión y solo confunden."""
    t = clf.tree_

    # Para cada nodo: ¿todo su subárbol predice siempre la misma clase? Si sí, cuál.
    clase_constante = {}

    def calcular_clase_constante(nodo):
        izq = t.children_left[nodo]
        if izq == -1:
            c = int(np.argmax(t.value[nodo]))
            clase_constante[nodo] = c
            return c
        der = t.children_right[nodo]
        c_izq = calcular_clase_constante(izq)
        c_der = calcular_clase_constante(der)
        c = c_izq if (c_izq is not None and c_izq == c_der) else None
        clase_constante[nodo] = c
        return c

    calcular_clase_constante(0)

    def es_hoja_efectiva(nodo):
        return t.children_left[nodo] == -1 or clase_constante[nodo] is not None

    # Elegir automáticamente cuántos decimales usar en los umbrales visibles (tras podar),
    # para que dos umbrales distintos nunca se muestren redondeados igual (ej. 2.55 y 2.53).
    umbrales = [t.threshold[n] for n in range(t.node_count)
                if t.children_left[n] != -1 and not es_hoja_efectiva(n)]
    decimales = 1
    for dec in range(1, 5):
        if len({round(u, dec) for u in umbrales}) == len(umbrales):
            decimales = dec
            break
    else:
        decimales = 4

    pos = {}
    contador = [0]

    def asignar_x(nodo, profundidad):
        if es_hoja_efectiva(nodo):
            x = contador[0]; contador[0] += 1
            pos[nodo] = (x, -profundidad)
            return x
        izq, der = t.children_left[nodo], t.children_right[nodo]
        xl = asignar_x(izq, profundidad + 1)
        xr = asignar_x(der, profundidad + 1)
        x = (xl + xr) / 2
        pos[nodo] = (x, -profundidad)
        return x

    asignar_x(0, 0)
    if colores is None:
        cmap = plt.cm.Set2
        colores = [cmap(i / max(len(class_names) - 1, 1)) for i in range(len(class_names))]

    for nodo, (x, y) in pos.items():
        if es_hoja_efectiva(nodo):
            clase = clase_constante[nodo] if clase_constante[nodo] is not None else int(np.argmax(t.value[nodo]))
            texto = class_names[clase]
            color = colores[clase]
        else:
            feat = feature_names[t.feature[nodo]]
            thr = t.threshold[nodo]
            texto = f"{feat} <= {thr:.{decimales}f}"
            color = "#e8e8e8"
        ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.28), 0.84, 0.56,
                                    boxstyle="round,pad=0.04", facecolor=color, edgecolor="black", linewidth=1))
        ax.text(x, y, texto, ha="center", va="center", fontsize=8.5)

    for nodo, (x0, y0) in pos.items():
        if not es_hoja_efectiva(nodo):
            izq, der = t.children_left[nodo], t.children_right[nodo]
            for hijo, etiqueta, dx in [(izq, "Sí", -0.15), (der, "No", 0.15)]:
                x1, y1 = pos[hijo]
                ax.plot([x0, x1], [y0 - 0.28, y1 + 0.28], color="gray", lw=1, zorder=0)
                ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, etiqueta, fontsize=7.5, color="dimgray")

    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.axis("off")


# ---------------------------------------------------------------
# Conexión a InfluxDB (solo Consumo_total)
# ---------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Consultando InfluxDB...")
def consultar_influx(url, token, org, bucket, rango_dias):
    from influxdb_client import InfluxDBClient

    # timeout explícito (ms): sin esto, una red lenta o credenciales erróneas
    # pueden dejar la consulta colgada mucho tiempo antes de fallar.
    client = InfluxDBClient(url=url, token=token, org=org, timeout=15_000)
    registros = []
    try:
        query_api = client.query_api()

        flux_query = f'''
        from(bucket: "{bucket}")
          |> range(start: -{rango_dias}d)
          |> filter(fn: (r) => r["_field"] == "Consumo_total")
          |> keep(columns: ["_time", "_value"])
        '''

        tables = query_api.query(flux_query, org=org)
        for table in tables:
            for record in table.records:
                registros.append({"timestamp": record.get_time(), "Consumo_total": record.get_value()})
    finally:
        # se cierra siempre, incluso si la consulta falla, para no dejar
        # conexiones huérfanas acumulándose en reintentos sucesivos.
        client.close()

    df = pd.DataFrame(registros)
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Calculando método del codo...")
def calcular_inercias_codo(consumo_values, k_max=8):
    X = consumo_values.reshape(-1, 1)
    inercias = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inercias.append(km.inertia_)
    return inercias


# ---------------------------------------------------------------
# Entrenamiento cacheado de modelos
# ---------------------------------------------------------------
# Streamlit re-ejecuta TODO el script (todas las pestañas, no solo la visible)
# en cada interacción con cualquier widget. Sin cachear, mover un slider en la
# pestaña de KNN también reentrena KMeans, SVM, el árbol y su malla de 40k
# puntos. @st.cache_resource evita eso: el modelo solo se reentrena si sus
# parámetros o los datos realmente cambiaron.

@st.cache_resource(show_spinner=False)
def entrenar_lineal(X_train, y_train):
    # X_train/y_train SÍ deben ir en la clave de caché (sin "_") para que un
    # cambio de datos invalide el modelo guardado; solo lo evitamos con "_"
    # en objetos que no queremos que rompan el caché (p.ej. un modelo ya entrenado).
    return LinearRegression().fit(X_train, y_train)


@st.cache_resource(show_spinner=False)
def entrenar_knn(X, y, k_vecinos):
    modelo = KNeighborsClassifier(n_neighbors=k_vecinos)
    modelo.fit(X, y)
    return modelo


@st.cache_resource(show_spinner=False)
def entrenar_svm(X, y, kernel, C, gamma_valor):
    modelo = SVC(kernel=kernel, C=C, gamma=gamma_valor, probability=True, random_state=42)
    modelo.fit(X, y)
    return modelo


@st.cache_resource(show_spinner=False)
def entrenar_arbol(X, y, profundidad, min_hoja):
    modelo = DecisionTreeClassifier(
        max_depth=profundidad, min_samples_leaf=min_hoja, random_state=42, class_weight="balanced"
    )
    modelo.fit(X, y)
    return modelo


@st.cache_resource(show_spinner=False)
def entrenar_kmeans(X, k):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo.fit(X)
    return modelo


# ---------------------------------------------------------------
# Sidebar: conexión
# ---------------------------------------------------------------

st.sidebar.header("🔌 Conexión a InfluxDB")

with st.sidebar.form("form_conexion_influx"):
    influx_url = st.text_input("URL", placeholder="https://us-east-1-1.aws.cloud2.influxdata.com/")
    influx_token = st.text_input("Token", type="password")
    influx_org = st.text_input("Organización")
    influx_bucket = st.text_input("Bucket", value="Consumo_elec")
    rango_dias = st.slider("Rango de datos (días)", 1, 5, 1)
    enviado = st.form_submit_button("Conectar y consultar")

if enviado:
    if not (influx_url and influx_token and influx_org and influx_bucket):
        st.sidebar.error("Completa todos los campos de conexión.")
    else:
        try:
            df_nuevo = consultar_influx(influx_url, influx_token, influx_org, influx_bucket, rango_dias)
            if df_nuevo.empty:
                st.sidebar.warning("La consulta no devolvió datos. Revisa el rango o el nombre del bucket.")
            else:
                st.session_state["df_consumo"] = df_nuevo
                st.sidebar.success(f"{len(df_nuevo)} registros cargados ✅")
        except Exception as e:
            st.sidebar.error(f"Error de conexión: {e}")

df = st.session_state.get("df_consumo")

if df is None:
    st.info("Configura la conexión a InfluxDB en la barra lateral y presiona **Conectar y consultar** para cargar los datos.")
    st.stop()

st.title("Machine Learning: Consumo Total vs Hora del Día")
st.caption("K-Means · KNN · SVM · Regresión Lineal — todo sobre la relación Consumo_total vs Hora")

# ---------------------------------------------------------------
# Única variable derivada del timestamp
# ---------------------------------------------------------------
# InfluxDB Cloud devuelve los timestamps en UTC. Colombia está en UTC-5 todo el año
# (sin horario de verano), así que convertimos antes de extraer la hora local;
# si no se hiciera esto, todo el patrón de consumo quedaría desplazado 5 horas.

_ts = pd.to_datetime(df["timestamp"])
if _ts.dt.tz is not None:
    _ts = _ts.dt.tz_convert("America/Bogota")
df["timestamp_local"] = _ts
df["hora"] = _ts.dt.hour
umbral = df["Consumo_total"].median()
df["consumo_alto"] = (df["Consumo_total"] > umbral).astype(int)

tab_datos, tab_lineal, tab_kmeans, tab_knn, tab_arbol, tab_svm, tab_comparacion = st.tabs(
    ["📊 Datos", "📈 Regresión Lineal", "🟢 K-Means", "🔵 KNN", "🌳 Árbol de Decisión", "🟠 SVM", "⚖️ Comparación"]
)


# TAB: DATOS
# ---------------------------------------------------------------
with tab_datos:
    st.subheader("Consumo total vs. Tiempo")

    fig_serie, ax_serie = plt.subplots(figsize=(14, 3.5))
    ax_serie.plot(df["timestamp_local"], df["Consumo_total"], linewidth=0.8)
    ax_serie.set_xlabel("Fecha (hora Colombia, UTC-5)")
    ax_serie.set_ylabel("Consumo total")
    ax_serie.set_title("Consumo total vs. Tiempo")
    fig_serie.autofmt_xdate()
    st.pyplot(fig_serie)
    plt.close(fig_serie)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df["hora"], df["Consumo_total"], alpha=0.3, s=15)
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Consumo total")
        ax.set_title("Consumo total vs. Hora del día")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        promedio_por_hora = df.groupby("hora")["Consumo_total"].mean()
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.bar(promedio_por_hora.index, promedio_por_hora.values)
        ax2.set_xlabel("Hora del día")
        ax2.set_ylabel("Consumo total promedio")
        ax2.set_title("Patrón de consumo promedio por hora")
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown(f"**Umbral de 'consumo alto'** (mediana): `{umbral:.2f}`")


# TAB: REGRESIÓN LINEAL
# ---------------------------------------------------------------
with tab_lineal:
    st.header("Regresión Lineal")
    st.markdown("¿Existe una tendencia lineal simple entre `hora` y `Consumo_total`?")

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        test_size_lin = st.slider("Proporción de test", 0.1, 0.5, 0.3, step=0.05, key="test_lin")

    X = df[["hora"]].values
    y = df["Consumo_total"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_lin, random_state=42)

    reg = entrenar_lineal(X_train, y_train)
    pred = reg.predict(X_test)

    with col_plot:
        orden = X_test[:, 0].argsort()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(df["hora"], df["Consumo_total"], alpha=0.3, s=15, label="Datos reales")
        ax.plot(X_test[orden], pred[orden], color="red", linewidth=2, label="Regresión Lineal")
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Consumo total")
        ax.set_title("Una línea recta vs. el patrón real")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2_score(y_test, pred):.3f}")
    c2.metric("MAE", f"{mean_absolute_error(y_test, pred):.3f}")
    c3.metric("RMSE", f"{mean_squared_error(y_test, pred) ** 0.5:.3f}")

    reg_full = entrenar_lineal(X, y)

    st.markdown("**Probar una predicción**")
    hora_pred_lin = st.slider("Hora del día", 0, 23, 12, key="hora_pred_lin")
    consumo_pred_lin = reg_full.predict([[hora_pred_lin]])[0]
    st.metric(f"Consumo estimado a las {hora_pred_lin}:00", f"{consumo_pred_lin:.1f}")

    with st.expander("📘 Conceptos clave"):
        st.markdown(
            """
            - Si el patrón diario tiene más de un pico (ej. mañana y noche), una sola línea
              recta **no puede** capturarlo bien — por eso el R² suele ser bajo aquí.
            - Esto motiva usar modelos más flexibles (K-Means, KNN, SVM) para el resto del análisis.
            """
        )


with tab_kmeans:
    st.header("K-Means — niveles de consumo (bajo, medio, alto)")
    st.markdown(
        """
        Agrupamos usando **solo `Consumo_total`** (no `hora` + `Consumo_total` juntos).
        Mezclar ambas variables con la misma escala tiende a producir clusters confusos,
        porque los valores atípicos de consumo dominan la distancia y la hora queda como
        una señal casi decorativa. Agrupando solo por consumo, los clusters representan
        **niveles reales** (bajo/medio/alto) — y usamos `hora` después, solo para interpretar.
        """
    )

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        k_elegido = st.slider("Número de clusters (k)", 2, 8, 3)
        mostrar_codo = st.checkbox("Mostrar método del codo", value=True)

    X_km = df[["Consumo_total"]].values

    kmeans = entrenar_kmeans(X_km, k_elegido)
    labels_raw = kmeans.predict(X_km)

    # Reordenar etiquetas para que 0 = consumo más bajo
    orden_clusters = pd.Series(df["Consumo_total"].values).groupby(labels_raw).mean().sort_values().index
    mapa_orden = {viejo: nuevo for nuevo, viejo in enumerate(orden_clusters)}
    df["cluster"] = pd.Series(labels_raw).map(mapa_orden).values

    with col_plot:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        scatter = ax.scatter(df["hora"], df["Consumo_total"], c=df["cluster"], cmap="viridis", s=25, alpha=0.8)
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Consumo total")
        ax.set_title(f"Niveles de consumo (k={k_elegido}) — coloreados por hora para interpretar")
        plt.colorbar(scatter, ax=ax, label="Nivel (0=bajo)")
        st.pyplot(fig)
        plt.close(fig)

    st.metric("Inercia", f"{kmeans.inertia_:.1f}")

    if mostrar_codo:
        inercias = calcular_inercias_codo(df["Consumo_total"].values)
        rango_k = range(1, len(inercias) + 1)
        fig_codo, ax_codo = plt.subplots(figsize=(6.5, 3.2))
        ax_codo.plot(list(rango_k), inercias, marker="o")
        ax_codo.axvline(k_elegido, color="red", linestyle="--", alpha=0.6, label="k elegido")
        ax_codo.set_xlabel("k")
        ax_codo.set_ylabel("Inercia")
        ax_codo.set_title("Método del codo (solo Consumo_total)")
        ax_codo.legend()
        st.pyplot(fig_codo)
        plt.close(fig_codo)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Rangos de consumo por nivel**")
        st.dataframe(
            df.groupby("cluster")["Consumo_total"].agg(["mean", "min", "max", "count"]).round(2),
            use_container_width=True,
        )
    with col_b:
        st.markdown("**¿A qué hora ocurre cada nivel?** (interpretación, no input del modelo)")
        st.dataframe(
            df.groupby("cluster")["hora"].agg(["mean", "min", "max"]).round(1),
            use_container_width=True,
        )

    st.markdown("**Distribución de horas dentro de cada nivel**")
    fig_hist, axes = plt.subplots(1, k_elegido, figsize=(4 * k_elegido, 3), sharey=True)
    if k_elegido == 1:
        axes = [axes]
    for cluster_id, ax in enumerate(axes):
        subset = df[df["cluster"] == cluster_id]
        ax.hist(subset["hora"], bins=24, range=(0, 24), color=plt.cm.viridis(cluster_id / max(k_elegido - 1, 1)))
        ax.set_title(f"Nivel {cluster_id} (n={len(subset)})")
        ax.set_xlabel("Hora")
    axes[0].set_ylabel("Frecuencia")
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.markdown("**Probar una predicción**")
    consumo_pred_km = st.number_input(
        "Valor de consumo", value=float(df["Consumo_total"].median()),
        min_value=0.0, step=0.1, key="consumo_pred_km"
    )
    nivel_crudo = kmeans.predict([[consumo_pred_km]])[0]
    nivel_pred = mapa_orden[nivel_crudo]
    st.metric(f"Consumo = {consumo_pred_km:.1f}", f"Nivel {nivel_pred}",
              help="0 = nivel de consumo más bajo, con k-1 = nivel más alto.")

    with st.expander("📘 Conceptos clave"):
        st.markdown(
            """
            - K-Means agrupa aquí **solo por nivel de consumo** — los rangos de consumo entre
              clusters no se solapan, por construcción.
            - La hora **no es input del modelo**, se usa solo para interpretar después: si el
              histograma de un nivel se concentra en ciertas horas, hay relación hora-consumo;
              si se ve disperso en las 24 horas, el nivel de consumo depende de otros factores.
            - Esto es más honesto que forzar a K-Means a usar hora y consumo juntos, lo cual
              puede producir "franjas horarias" que en realidad no son consistentes.
            """
        )



def plot_clasificacion_por_hora(modelo, df, umbral, titulo):
    horas_grid = np.arange(0, 24, 0.1).reshape(-1, 1)
    pred_grid = modelo.predict(horas_grid)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(len(horas_grid) - 1):
        color = "#fde0dd" if pred_grid[i] == 0 else "#c6dbef"
        ax.axvspan(horas_grid[i, 0], horas_grid[i + 1, 0], color=color, alpha=0.6, linewidth=0)

    ax.scatter(df["hora"], df["Consumo_total"], c=df["consumo_alto"], cmap="coolwarm",
               edgecolor="k", s=20, alpha=0.8)
    ax.axhline(umbral, color="black", linestyle="--", alpha=0.5, label="Umbral (mediana)")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Consumo total")
    ax.set_title(titulo)
    ax.legend()
    return fig


# ---------------------------------------------------------------
# TAB: KNN
# ---------------------------------------------------------------
with tab_knn:
    st.header("KNN — clasificar consumo alto/bajo según horas vecinas")
    st.markdown("Para predecir si una hora tendrá consumo alto, KNN mira qué pasó en las horas más parecidas.")

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        k_vecinos = st.slider("Número de vecinos (k)", 1, 30, 5)
        test_size_knn = st.slider("Proporción de test", 0.1, 0.5, 0.3, step=0.05, key="test_knn")

    X_clas = df[["hora"]].values
    y_clas = df["consumo_alto"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_clas, y_clas, test_size=test_size_knn, random_state=42, stratify=y_clas
    )

    knn = entrenar_knn(X_train, y_train, k_vecinos)
    pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, pred_knn)

    knn_full = entrenar_knn(X_clas, y_clas, k_vecinos)

    with col_plot:
        fig = plot_clasificacion_por_hora(knn_full, df, umbral, f"KNN (k={k_vecinos}): horas alto (azul) vs bajo (rosa)")
        st.pyplot(fig)
        plt.close(fig)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc_knn:.2%}")
    with c2:
        cm = confusion_matrix(y_test, pred_knn)
        fig_cm, ax_cm = plt.subplots(figsize=(3.2, 3.2))
        ConfusionMatrixDisplay(cm, display_labels=["Bajo", "Alto"]).plot(ax=ax_cm, cmap="Blues", colorbar=False)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    st.markdown("**Probar una predicción**")
    hora_pred_knn = st.slider("Hora del día", 0, 23, 12, key="hora_pred_knn")
    etiqueta_knn = knn_full.predict([[hora_pred_knn]])[0]
    proba_knn = knn_full.predict_proba([[hora_pred_knn]])[0]
    texto_knn = "Alto" if etiqueta_knn == 1 else "Bajo"
    confianza_knn = proba_knn[etiqueta_knn] * 100
    st.metric(f"Predicción para las {hora_pred_knn}:00", f"Consumo {texto_knn}", f"{confianza_knn:.0f}% de probabilidad")

    with st.expander("📘 Conceptos clave"):
        st.markdown(
            """
            - Con `k` bajo, el mapa de franjas puede verse muy irregular (sensible al ruido).
            - Con `k` alto, las franjas se suavizan, pero pueden perder detalle si son demasiado altas.
            - El fondo de color muestra qué predice el modelo para cada hora — si aparecen dos
              franjas azules separadas, el modelo capturó bien los dos picos de consumo.
            """
        )

# ---------------------------------------------------------------
# TAB: ÁRBOL DE DECISIÓN
# ---------------------------------------------------------------
with tab_arbol:
    st.header("Árbol de Decisión — detección de consumo anómalo")
    st.markdown(
        "Dado la **hora** y el **valor de consumo**, el árbol dice si ese registro es "
        "normal o anómalo para esa hora del día — más realista en analítica energética "
        "que solo clasificar alto/bajo contra la mediana global."
    )

   
    profundidad = 3          # suficiente para separar anomalías por encima y por debajo
    min_hoja_pct = 3.0       # cada hoja necesita al menos 3% de los datos para formarse
    test_size_arbol = 0.3

    sensibilidad = st.slider(
        "Sensibilidad (desviaciones estándar)", 1.5, 4.0, 2.5, step=0.5,
        help="Un registro se etiqueta como anómalo si se aleja más de este número de "
             "desviaciones estándar del consumo promedio esperado para esa hora del día."
    )

    min_hoja = max(1, int(len(df) * min_hoja_pct / 100))

  
   
    patron_media_hora = df.groupby("hora")["Consumo_total"].mean().reindex(range(24)).interpolate().bfill().ffill()
    patron_std_hora = df.groupby("hora")["Consumo_total"].std().reindex(range(24)).interpolate().bfill().ffill()
    patron_std_hora = patron_std_hora.replace(0, patron_std_hora[patron_std_hora > 0].min())

    def calcular_desviacion(hora_arr, consumo_arr):
        """Cuánto se aleja un consumo del promedio esperado para su hora, en unidades de desviación estándar."""
        media = patron_media_hora.reindex(np.asarray(hora_arr)).values
        std = patron_std_hora.reindex(np.asarray(hora_arr)).values
        return (np.asarray(consumo_arr) - media) / std

    df["desviacion"] = calcular_desviacion(df["hora"], df["Consumo_total"])
    df["anomalo"] = (df["desviacion"].abs() > sensibilidad).fillna(False).astype(int)

    n_anomalos = int(df["anomalo"].sum())
    st.caption(
        f"Con esta sensibilidad: **{n_anomalos} de {len(df)}** registros marcados como anómalos "
        f"({n_anomalos/len(df):.1%}). Cada hoja del árbol necesitará al menos **{min_hoja} datos** "
        f"({min_hoja_pct:.1f}%) para poder formarse."
    )

    if n_anomalos == 0 or n_anomalos == len(df):
        st.warning(
            "Con esta sensibilidad no hay variedad suficiente entre las dos clases "
            "(todo quedó como normal, o todo como anómalo). Ajusta el slider de sensibilidad."
        )
    else:
        
        X_arbol = df[["desviacion"]].values
        y_arbol = df["anomalo"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_arbol, y_arbol, test_size=test_size_arbol, random_state=42, stratify=y_arbol
        )

       
        arbol = entrenar_arbol(X_train, y_train, profundidad, min_hoja)
        pred_arbol = arbol.predict(X_test)

        arbol_full = entrenar_arbol(X_arbol, y_arbol, profundidad, min_hoja)

        acc_arbol = accuracy_score(y_test, pred_arbol)
        cm = confusion_matrix(y_test, pred_arbol, labels=[0, 1])
        recall_anomalo = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else float("nan")

       
        h_min, h_max = -0.5, 23.5
        c_min, c_max = df["Consumo_total"].min() * 0.95, df["Consumo_total"].max() * 1.05
        hh, cc = np.meshgrid(np.linspace(h_min, h_max, 200), np.linspace(c_min, c_max, 200))
        hh_redondeada = np.clip(np.round(hh), 0, 23)
        desv_grid = calcular_desviacion(hh_redondeada.ravel(), cc.ravel())
        Z = arbol_full.predict(desv_grid.reshape(-1, 1)).reshape(hh.shape)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.contourf(hh, cc, Z, alpha=0.25, cmap="coolwarm")
        ax.scatter(df["hora"], df["Consumo_total"], c=df["anomalo"], cmap="coolwarm",
                  edgecolor="k", s=20, alpha=0.85)
        ax.plot(range(24), patron_media_hora.values, color="black", linestyle="--", linewidth=1, label="Promedio por hora")
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Consumo total")
        ax.set_title(f"Normal (azul) vs. Anómalo (rojo)")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{acc_arbol:.2%}")
        c2.metric("Recall en anomalías", f"{recall_anomalo:.2%}",
                  help="De todas las anomalías reales en el set de prueba, ¿qué porcentaje detectó el árbol?")

        with st.expander("Ver matriz de confusión y reporte completo"):
            fig_cm, ax_cm = plt.subplots(figsize=(3.2, 3.2))
            ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anómalo"]).plot(
                ax=ax_cm, cmap="Reds", colorbar=False
            )
            st.pyplot(fig_cm)
            plt.close(fig_cm)

      
        st.markdown("**Árbol de decisión:**")
        fig_tree, ax_tree = plt.subplots(figsize=(9, 2 + profundidad))
        dibujar_arbol_simple(arbol_full, ["desviación (σ)"], ["Normal", "Anómalo"], ax_tree)
        plt.tight_layout()
        st.pyplot(fig_tree)
        plt.close(fig_tree)
        st.caption(
            "El árbol corta directamente sobre la **desviación** (qué tan lejos está el consumo "
            "del promedio esperado para esa hora, en desviaciones estándar) — no sobre hora y "
            "consumo por separado. Además, cualquier corte que no cambie la clase predicha a "
            "ningún lado (algo que CART sí puede generar, persiguiendo pureza sin cambiar la "
            "decisión) se colapsa en una sola hoja, para que cada corte visible signifique algo."
        )

       
        st.markdown("**Probar una predicción**")
        pc1, pc2 = st.columns(2)
        hora_consulta = pc1.slider("Hora del día", 0, 23, 12, key="hora_consulta_arbol")
        consumo_consulta = pc2.number_input(
            "Valor de consumo", value=float(df["Consumo_total"].median()),
            min_value=0.0, step=0.1, key="consumo_consulta_arbol"
        )

        desviacion_consulta = calcular_desviacion([hora_consulta], [consumo_consulta])[0]
        entrada = [[desviacion_consulta]]
        etiqueta_pred = arbol_full.predict(entrada)[0]
        proba_pred = arbol_full.predict_proba(entrada)[0]
        clases = list(arbol_full.classes_)
        confianza = proba_pred[clases.index(etiqueta_pred)] * 100
        texto_etiqueta = "Anómalo ⚠️" if etiqueta_pred == 1 else "Normal ✅"

        st.metric(
            f"Predicción para hora={hora_consulta}h, consumo={consumo_consulta:.1f}",
            texto_etiqueta,
            f"{confianza:.0f}% de probabilidad",
        )

        with st.expander("📘 Conceptos clave"):
            st.markdown(
                """
                - La etiqueta "anómalo" no viene de una fuente externa: se calcula comparando
                  cada registro contra el **promedio y desviación estándar de su propia hora**
                  (mismo principio detrás de un control tipo CUSUM, simplificado a un umbral fijo).
                - Al árbol **no** le damos hora y consumo por separado — le damos la **desviación
                  ya calculada** (cuántas desviaciones estándar se aleja del promedio esperado
                  para esa hora). Si le diéramos hora y consumo en bruto, el árbol tendría que
                  reconstruir a fuerza de cortes rectangulares una frontera que en realidad es
                  curva (la banda "normal" sube y baja según la hora), y terminaría con cortes
                  anidados y repetidos, difíciles de leer.
                - Por eso ahora el árbol es pequeño y directo: prácticamente reaprende la misma
                  regla que usamos para etiquetar (`|desviación| > sensibilidad`) — lo cual es
                  esperado y es una buena señal de que el modelo capturó justo lo que se buscaba,
                  no ruido.
                - Si quieres que el árbol descubra algo **más allá** de la regla estadística, la
                  etiqueta tendría que venir de otro lado (ej. fallas reportadas manualmente,
                  mantenimientos, cortes de producción) en vez de derivarse del mismo z-score.
                - `class_weight="balanced"` evita que el árbol ignore las anomalías por ser pocas
                  — sin esto, un árbol perezoso podría llegar a >90% de accuracy con solo predecir
                  "Normal" siempre, y el `Recall en anomalías` sería 0%. Por eso esa métrica importa
                  más que el accuracy en este problema.
                - Aumentar la sensibilidad (más desviaciones estándar) genera menos anomalías, pero
                  más "seguras"; bajarla genera más alertas, con más falsos positivos.
                """
            )

# ---------------------------------------------------------------
# TAB: SVM
# ---------------------------------------------------------------
with tab_svm:
    st.header("SVM — mismo problema, fronteras más flexibles")
    st.markdown("Comparación directa contra KNN, usando el mismo par de franjas alto/bajo.")

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        C = st.select_slider("C", options=[0.01, 0.1, 1, 10, 100], value=1)
        gamma_valor = st.select_slider("gamma", options=["scale", 0.01, 0.1, 1, 10], value="scale")
        test_size_svm = st.slider("Proporción de test", 0.1, 0.5, 0.3, step=0.05, key="test_svm")

    X_svm = df[["hora"]].values
    y_svm = df["consumo_alto"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_svm, y_svm, test_size=test_size_svm, random_state=42, stratify=y_svm
    )

    modelo_svm = entrenar_svm(X_train, y_train, kernel, C, gamma_valor)
    pred_svm = modelo_svm.predict(X_test)
    acc_svm = accuracy_score(y_test, pred_svm)

    svm_full = entrenar_svm(X_svm, y_svm, kernel, C, gamma_valor)

    with col_plot:
        fig = plot_clasificacion_por_hora(svm_full, df, umbral, f"SVM ({kernel}): horas alto (azul) vs bajo (rosa)")
        st.pyplot(fig)
        plt.close(fig)

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc_svm:.2%}")
    c2.metric("Vectores de soporte", int(modelo_svm.support_vectors_.shape[0]))

    st.markdown("**Probar una predicción**")
    hora_pred_svm = st.slider("Hora del día", 0, 23, 12, key="hora_pred_svm")
    etiqueta_svm = svm_full.predict([[hora_pred_svm]])[0]
    proba_svm = svm_full.predict_proba([[hora_pred_svm]])[0]
    texto_svm = "Alto" if etiqueta_svm == 1 else "Bajo"
    confianza_svm = proba_svm[etiqueta_svm] * 100
    st.metric(f"Predicción para las {hora_pred_svm}:00", f"Consumo {texto_svm}", f"{confianza_svm:.0f}% de probabilidad")

    with st.expander("📘 Conceptos clave"):
        st.markdown(
            """
            - Con kernel `linear`, SVM solo puede definir **una** franja de corte — si hay dos
              picos de consumo separados, fallará en capturar ambos correctamente.
            - Con kernel `rbf`, SVM puede definir varias franjas, similar a KNN.
            - Compara el accuracy y la forma del mapa aquí contra la pestaña de KNN.
            """
        )

# ---------------------------------------------------------------
# TAB: COMPARACIÓN
# ---------------------------------------------------------------
with tab_comparacion:
    st.header("Comparación de modelos")
    st.markdown(
        """
        | Algoritmo | Tipo de problema | ¿Usa etiquetas? | Qué responde en este caso |
        |---|---|---|---|
        | Regresión Lineal | Regresión (valor continuo) | Sí | ¿Cuánto será el consumo según la hora? (relación simple) |
        | K-Means | Clustering | No | ¿Qué franjas horarias de consumo existen, sin definirlas de antemano? |
        | KNN | Clasificación | Sí | ¿Esta hora tendrá consumo alto o bajo, según horas parecidas? |
        | Árbol de Decisión | Clasificación | Sí | Dado hora + valor de consumo, ¿es un registro anómalo para esa hora? |
        | SVM | Clasificación | Sí | ¿Esta hora tendrá consumo alto o bajo? (frontera global, más flexible) |
        """
    )
    st.info("Ajusta parámetros en cada pestaña y vuelve aquí para comparar tus propias métricas anotadas.")

st.divider()
st.caption("App educativa · ML sobre Consumo_total vs Hora (InfluxDB) · Regresión Lineal, K-Means, KNN, Árbol de Decisión, SVM")