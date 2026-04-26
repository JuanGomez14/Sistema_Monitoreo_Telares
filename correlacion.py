"""
correlacion.py
=============
Motor de correlación y dashboard de alertas.

Une las señales de vibración (FFT_analysys.py) con los resultados de visión (Vision_analysis.py)
para generar alertas de alta confianza cuando ambas fuentes coinciden.

Niveles de alerta:
    0 - NORMAL       : sin anomalías en ninguna fuente
    1 - ALERTA_VIB   : solo vibración anómala
    2 - ALERTA_VIS   : solo defecto visual detectado
    3 - ALERTA_ALTA  : ambas fuentes confirman fallo → acción inmediata

Autor: Tu nombre aquí
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# Parámetros del correlador
# ─────────────────────────────────────────────

# Umbrales de decisión
VIB_FAULT_THRESH    = 0.3    # fracción de ventanas con alerta para declarar fallo
VISION_FAULT_THRESH = 0.35   # defect_score mínimo para declarar defecto visual

# Ventana de correlación temporal (segundos)
# Las vibraciones y las imágenes se sincronizan cada CORR_WINDOW segundos
CORR_WINDOW_SEC     = 1.0

# Colores del dashboard
COLORS = {
    "normal":      "#00d4ff",
    "Solo_vib":    "#ffa502",
    "Solo_vis":    "#a29bfe",
    "Alerta_alta":  "#ff4757",
    "bg":          "#0f0f1a",
    "panel":       "#1a1a2e",
    "text":        "#ffffff",
    "subtext":     "#aaaaaa",
    "grid":        "#222244",
}

ALERT_LABELS = {
    0: "NORMAL",
    1: "ALERTA VIBRACIÓN",
    2: "ALERTA VISUAL",
    3: "ALERTA ALTA",
}

ALERT_COLORS = {
    0: COLORS["normal"],
    1: COLORS["Solo_vib"],
    2: COLORS["Solo_vis"],
    3: COLORS["Alerta_alta"],
}


# ─────────────────────────────────────────────
# Carga y preparación de datos
# ─────────────────────────────────────────────

def load_vibration_report(analysis_dir: Path,
                           fault_type: str = "frequency") -> pd.DataFrame:
    """
    Carga el reporte de análisis FFT generado.
    Retorna DataFrame con columnas de tiempo y flags de alerta.
    """
    path = analysis_dir / f"report_{fault_type}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encontrado: {path}")

    df = pd.read_csv(path)
    print(f" Vibración cargada: {path.name}  ({len(df)} ventanas)")
    return df


def load_vision_results(vision_dir: Path,
                         category: str = "grid") -> pd.DataFrame:
    """
    Carga los resultados de visión generados.
    """
    path = vision_dir / f"vision_results_{category}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encontrado: {path}")

    df = pd.read_csv(path)
    print(f" Visión cargada: {path.name}  ({len(df)} imágenes)")
    return df


def simulate_timeline(vib_df: pd.DataFrame,
                       vision_df: pd.DataFrame,
                       duration_sec: float = 60.0,
                       images_per_sec: float = 0.5) -> pd.DataFrame:
    """
    Construye una línea de tiempo unificada combinando señales de
    vibración e imágenes.

    En un sistema real, las marcas de tiempo vendrían de los sensores.
    Aquí las simulamos distribuyendo los datos en el tiempo.

    Args:
        vib_df        : reporte FFT con ventanas de vibración
        vision_df     : resultados de visión con defect_score
        duration_sec  : duración total de la simulación
        images_per_sec: frecuencia de captura de imágenes

    Returns:
        DataFrame con una fila por segundo, columnas de ambas fuentes
        y nivel de alerta calculado.
    """
    # ── Eje de tiempo ──
    timestamps = np.arange(0, duration_sec, CORR_WINDOW_SEC)
    n_steps    = len(timestamps)

    # ── Señal de vibración → distribuir ventanas en el tiempo ──
    # Repetir/recortar el reporte para cubrir toda la duración
    vib_repeated = pd.concat(
        [vib_df] * (n_steps // len(vib_df) + 1), ignore_index=True
    ).head(n_steps)

    vib_fault_series = vib_repeated["any_fault"].values.astype(float)
    vib_score_series = vib_repeated["anomaly_pwr_80hz"].values

    # ── Señal de visión → distribuir imágenes en el tiempo ──
    # Una imagen cada 1/images_per_sec segundos
    vision_scores  = np.zeros(n_steps)
    vision_defects = np.zeros(n_steps, dtype=bool)

    n_images    = len(vision_df)
    img_indices = (timestamps * images_per_sec).astype(int) % n_images

    for i, idx in enumerate(img_indices):
        vision_scores[i]  = vision_df.iloc[idx]["defect_score"]
        vision_defects[i] = vision_df.iloc[idx]["is_defect_pred"]

    # ── Construir DataFrame unificado ──
    timeline = pd.DataFrame({
        "timestamp_s":    timestamps,
        "vib_fault":      vib_fault_series,
        "vib_score":      vib_score_series,
        "vision_score":   vision_scores,
        "vision_fault":   vision_defects.astype(int),
    })

    # ── Calcular nivel de alerta ──
    timeline["alert_level"] = timeline.apply(_compute_alert_level, axis=1)
    timeline["alert_label"] = timeline["alert_level"].map(ALERT_LABELS)

    return timeline


def _compute_alert_level(row: pd.Series) -> int:
    """
    Regla de decisión para el nivel de alerta.

    Lógica:
        vib y vis  → 3 (ALTA)
        Solo vib     → 1
        Solo vis     → 2
        ninguno      → 0
    """
    has_vib = bool(row["vib_fault"])
    has_vis = bool(row["vision_fault"])

    if has_vib and has_vis:
        return 3
    elif has_vib:
        return 1
    elif has_vis:
        return 2
    else:
        return 0


# ─────────────────────────────────────────────
# Generador de reporte de alertas
# ─────────────────────────────────────────────

def generate_alert_report(timeline: pd.DataFrame,
                           output_dir: Path,
                           label: str = "simulation") -> pd.DataFrame:
    """
    Extrae los eventos de alerta de la línea de tiempo y genera
    un reporte de incidentes.

    Agrupa alertas consecutivas del mismo tipo en un solo evento.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resumen de distribución de alertas
    dist = timeline["alert_level"].value_counts().sort_index()

    print(f"\n  {'─'*45}")
    print(f"  DISTRIBUCIÓN DE ALERTAS — {label.upper()}")
    print(f"  {'─'*45}")
    for level, count in dist.items():
        pct   = count / len(timeline) * 100
        label_str = ALERT_LABELS[level]
        bar   = " " * int(pct / 2)
        print(f"  {label_str:20s} : {count:4d} ({pct:5.1f}%)  {bar}")

    # Extraer eventos de alerta alta (nivel 3)
    high_alerts = timeline[timeline["alert_level"] == 3].copy()
    print(f"\n  Eventos ALERTA ALTA : {len(high_alerts)}")
    print(f"  Tiempo total en alerta alta: "
          f"{len(high_alerts) * CORR_WINDOW_SEC:.1f}s "
          f"/ {len(timeline) * CORR_WINDOW_SEC:.0f}s total")

    # Guardar timeline completa
    timeline_path = output_dir / f"timeline_{label}.csv"
    timeline.to_csv(timeline_path, index=False)
    print(f"\n  Timeline guardada: {timeline_path}")

    return high_alerts


# ─────────────────────────────────────────────
# Dashboard principal
# ─────────────────────────────────────────────

def plot_dashboard(timeline: pd.DataFrame,
                   output_dir: Path,
                   label: str = "simulation") -> None:
    """
    Genera el dashboard completo de monitoreo con 5 paneles:

    Panel 1 (top, ancho): Línea de tiempo de alertas con colores
    Panel 2: Score de vibración vs tiempo
    Panel 3: Score de visión vs tiempo
    Panel 4: Nivel de alerta vs tiempo (barras)
    Panel 5: Distribución de alertas (pie chart)
    """
    t = timeline["timestamp_s"].values

    fig = plt.figure(figsize=(18, 12), facecolor=COLORS["bg"])
    fig.suptitle(
        " TELAR MONITOR — Dashboard de Monitoreo Industrial\n"
        f"Simulación: {label.upper()}  |  "
        f"Duración: {t[-1]:.0f}s  |  "
        f"Alertas altas: {(timeline['alert_level']==3).sum()}",
        color=COLORS["text"], fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.35,
                           top=0.92, bottom=0.06)

    # ── Panel 1: Timeline de alertas (ancho completo) ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(COLORS["panel"])
    ax1.set_title("Línea de Tiempo — Niveles de Alerta",
                  color=COLORS["text"], fontsize=10, pad=6)

    for level in [0, 1, 2, 3]:
        mask = timeline["alert_level"] == level
        ax1.fill_between(t, 0, 1, where=mask.values,
                         color=ALERT_COLORS[level], alpha=0.6,
                         label=ALERT_LABELS[level])

    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Tiempo (s)", color=COLORS["subtext"], fontsize=8)
    ax1.set_yticks([])
    ax1.tick_params(colors=COLORS["subtext"])
    for sp in ax1.spines.values(): sp.set_color(COLORS["grid"])
    legend = ax1.legend(loc="upper right", fontsize=8,
                        facecolor=COLORS["panel"], labelcolor=COLORS["text"],
                        ncol=4)

    # ── Panel 2: Score de vibración ──
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor(COLORS["panel"])
    ax2.set_title("Score de Vibración — Potencia @ 80Hz",
                  color=COLORS["text"], fontsize=10, pad=6)
    ax2.plot(t, timeline["vib_score"], color=COLORS["Solo_vib"],
             linewidth=0.8, alpha=0.9)
    ax2.fill_between(t, 0, timeline["vib_score"],
                     color=COLORS["Solo_vib"], alpha=0.15)
    # Sombrear alertas altas
    high_mask = timeline["alert_level"] == 3
    ax2.fill_between(t, 0, timeline["vib_score"].max(),
                     where=high_mask.values,
                     color=COLORS["Alerta_alta"], alpha=0.12)
    ax2.set_xlabel("Tiempo (s)", color=COLORS["subtext"], fontsize=8)
    ax2.set_ylabel("Potencia", color=COLORS["subtext"], fontsize=8)
    ax2.tick_params(colors=COLORS["subtext"])
    for sp in ax2.spines.values(): sp.set_color(COLORS["grid"])
    ax2.grid(True, color=COLORS["grid"], linewidth=0.4)

    # ── Panel 3: Score visual ──
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.set_facecolor(COLORS["panel"])
    ax3.set_title("Score Visual — Defect Score por Imagen",
                  color=COLORS["text"], fontsize=10, pad=6)
    ax3.plot(t, timeline["vision_score"], color=COLORS["Solo_vis"],
             linewidth=0.8, alpha=0.9)
    ax3.fill_between(t, 0, timeline["vision_score"],
                     color=COLORS["Solo_vis"], alpha=0.15)
    ax3.fill_between(t, 0, timeline["vision_score"].max(),
                     where=high_mask.values,
                     color=COLORS["Alerta_alta"], alpha=0.12)
    ax3.axhline(VISION_FAULT_THRESH, color=COLORS["Alerta_alta"],
                linewidth=1, linestyle="--", alpha=0.7,
                label=f"Umbral ({VISION_FAULT_THRESH})")
    ax3.set_xlabel("Tiempo (s)", color=COLORS["subtext"], fontsize=8)
    ax3.set_ylabel("Defect Score", color=COLORS["subtext"], fontsize=8)
    ax3.tick_params(colors=COLORS["subtext"])
    ax3.legend(fontsize=7, facecolor=COLORS["panel"],
               labelcolor=COLORS["text"])
    for sp in ax3.spines.values(): sp.set_color(COLORS["grid"])
    ax3.grid(True, color=COLORS["grid"], linewidth=0.4)

    # ── Panel 4: Nivel de alerta (barras verticales) ──
    ax4 = fig.add_subplot(gs[1:, 2])
    ax4.set_facecolor(COLORS["panel"])
    ax4.set_title("Distribución\nde Alertas",
                  color=COLORS["text"], fontsize=10, pad=6)

    dist    = timeline["alert_level"].value_counts().sort_index()
    levels  = [ALERT_LABELS[i] for i in dist.index]
    counts  = dist.values
    colors  = [ALERT_COLORS[i] for i in dist.index]
    pcts    = counts / counts.sum() * 100

    bars = ax4.barh(levels, counts, color=colors, alpha=0.85,
                    edgecolor=COLORS["bg"], linewidth=0.5)

    for bar, pct, count in zip(bars, pcts, counts):
        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{count} ({pct:.1f}%)",
                 va="center", ha="left",
                 color=COLORS["text"], fontsize=8)

    ax4.set_xlabel("Cantidad de ventanas", color=COLORS["subtext"], fontsize=8)
    ax4.tick_params(colors=COLORS["subtext"])
    for sp in ax4.spines.values(): sp.set_color(COLORS["grid"])
    ax4.grid(True, color=COLORS["grid"], linewidth=0.4, axis="x")

    # ── Guardar ──
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"dashboard_{label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Dashboard guardado: {out_path}")


# ─────────────────────────────────────────────
# Método principal
# ─────────────────────────────────────────────

def run_correlation(analysis_dir: Path = Path("data/analysis"),
                    vision_dir:   Path = Path("data/vision_output"),
                    output_dir:   Path = Path("data/dashboard"),
                    fault_type:   str  = "frequency",
                    category:     str  = "grid") -> pd.DataFrame:
    """
    Método completo de correlación:
        1. Carga reportes de vibración y visión
        2. Construye línea de tiempo unificada
        3. Calcula niveles de alerta por correlación
        4. Genera reporte de eventos
        5. Genera dashboard visual
    """
    print("\n" + "═" * 58)
    print("  TELAR MONITOR — Motor de Correlación")
    print("═" * 58)
    label = f"{fault_type}_vs_{category}"

    # 1. Cargar datos
    print("\n Cargando datos...")
    vib_df    = load_vibration_report(analysis_dir, fault_type)
    vision_df = load_vision_results(vision_dir, category)

    # 2. Construir timeline
    print("\n Construyendo línea de tiempo...")
    timeline = simulate_timeline(vib_df, vision_df,
                                  duration_sec=120.0,
                                  images_per_sec=0.5)
    print(f" Timeline: {len(timeline)} pasos de {CORR_WINDOW_SEC}s")

    # 3. Reporte de alertas
    print("\n Generando reporte de alertas...")
    high_alerts = generate_alert_report(timeline, output_dir, label)

    # 4. Dashboard
    print("\n Generando dashboard...")
    plot_dashboard(timeline, output_dir, label)

    print("\n" + "═" * 58)
    print("  CORRELACIÓN COMPLETADA")
    print("═" * 58 + "\n")

    return timeline


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Combinaciones a analizar
    combinations = [
        ("frequency", "grid"),
        ("spike",     "grid"),
        ("drift",     "grid"),
    ]

    for fault, category in combinations:
        print(f"\n{'━'*58}")
        print(f"  Combinación: {fault.upper()} + {category.upper()}")
        print(f"{'━'*58}")
        run_correlation(
            analysis_dir = Path("data/analysis"),
            vision_dir   = Path("data/vision_output"),
            output_dir   = Path("data/dashboard"),
            fault_type   = fault,
            category     = category,
        )