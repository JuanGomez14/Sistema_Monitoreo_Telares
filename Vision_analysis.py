"""
vision_analysys.py
==================
Visión artificial con OpenCV para detección de defectos en tela.

Método por imagen:
    1. Carga y preprocesamiento  (resize, grayscale, normalización)
    2. Análisis de textura       (uniformidad del patrón)
    3. Detección de bordes       (Canny + Sobel)
    4. Análisis de contornos     (tamaño, forma, densidad de anomalías)
    5. Score de defecto          (0.0 = perfecto, 1.0 = defecto severo)
    6. Reporte con Pandas        (etiqueta, score, tipo detectado)

Compatible con estructura MVTec AD:
    <categoria>/
        train/good/
        test/good/
        test/<defecto_1>/
        test/<defecto_2>/
        ...

Autor: Juan de Jesus Gomez Lopez
Elaborado: Abril 2026
"""

import cv2
from matplotlib import patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Parámetros del analizador
# ─────────────────────────────────────────────

# Tamaño estándar al que redimensionamos todas las imágenes
IMG_SIZE        = (256, 256)

# Parámetros Canny (detección de bordes)
CANNY_LOW       = 50
CANNY_HIGH      = 150

# Parámetros de decisión
# Un contorno es "sospechoso" si su área supera este % del total de la imagen
MIN_CONTOUR_AREA_PCT  = 0.001   # 0.1% del área total
MAX_CONTOUR_AREA_PCT  = 0.30    # >30% → probablemente borde de imagen, no defecto

# Umbral de score para declarar defecto
DEFECT_SCORE_THRESH   = 0.42


# ─────────────────────────────────────────────
# Preprocesamiento
# ─────────────────────────────────────────────

def preprocess(img_path: Path) -> dict:
    """
    Carga una imagen y genera todas las versiones que usaremos en el análisis.

    Returns dict con:
        original   : BGR original
        gray       : escala de grises
        blurred    : blur gaussiano (reduce ruido)
        normalized : float32 en rango [0,1]
        resized    : BGR redimensionado a IMG_SIZE
    """
    # Leer imagen en color
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar: {img_path}")

    # Redimensionar a tamaño estándar
    resized = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Escala de grises — la mayoría de análisis de textura trabaja aquí
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Blur gaussiano 5x5 — reduce ruido sin destruir bordes reales
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Normalizado a float [0,1] para operaciones matemáticas
    normalized = resized.astype(np.float32) / 255.0

    return {
        "original":   img_bgr,
        "resized":    resized,
        "gray":       gray,
        "blurred":    blurred,
        "normalized": normalized,
    }


# ─────────────────────────────────────────────
# Análisis de textura GLCM simplificado 
# (Grey-Level Co-occurrence Matrix o Matriz de Coocurrencia de Niveles de Gris)
# ─────────────────────────────────────────────

def analyze_texture(gray: np.ndarray) -> dict:
    """
    Mide la uniformidad del patrón de la tela.

    Una tela normal tiene textura REGULAR y predecible.
    Un defecto rompe esa regularidad.

    Métricas calculadas:
        - std_local   : desviación estándar de parches locales
                        (alta variabilidad → posible defecto)
        - uniformity  : qué tan uniforme es el brillo
                        (baja uniformidad → posible defecto)
        - entropy     : entropía de la imagen
                        (alta entropía → patrón irregular)
    """
    h, w = gray.shape
    patch_size = 32   # analizamos parches de 32x32 píxeles

    local_stds = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size]
            local_stds.append(patch.std())

    local_stds = np.array(local_stds)

    # Uniformidad: baja si hay grandes diferencias entre parches
    uniformity = 1.0 - (local_stds.std() / (local_stds.mean() + 1e-6))
    uniformity = float(np.clip(uniformity, 0.0, 1.0))

    # Entropía del histograma
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

    # Std promedio de parches locales (normalizado por rango de grises)
    std_local = float(local_stds.mean() / 128.0)

    return {
        "std_local":  round(std_local, 5),
        "uniformity": round(uniformity, 5),
        "entropy":    round(entropy, 5),
    }


# ─────────────────────────────────────────────
# Detección de bordes (Canny + Sobel)
# ─────────────────────────────────────────────

def detect_edges(blurred: np.ndarray) -> dict:
    """
    Aplica dos detectores de bordes complementarios.

    Canny  → bordes nítidos, bueno para huecos y cortes
    Sobel  → gradiente en X e Y, bueno para irregularidades de trama

    Returns:
        canny_edges    : imagen binaria de bordes Canny
        sobel_mag      : magnitud del gradiente Sobel
        edge_density   : % de píxeles que son borde (indica complejidad)
        sobel_mean     : intensidad media del gradiente
    """
    # Canny — detecta bordes como contornos binarios
    canny_edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # Sobel — calcula el gradiente de intensidad en cada dirección
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalizar Sobel a [0, 255]
    sobel_norm = cv2.normalize(sobel_mag, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)

    total_pixels = blurred.shape[0] * blurred.shape[1]
    edge_density = float(np.count_nonzero(canny_edges)) / total_pixels
    sobel_mean   = float(sobel_mag.mean())

    return {
        "canny_edges": canny_edges,
        "sobel_mag":   sobel_norm,
        "edge_density": round(edge_density, 5),
        "sobel_mean":   round(sobel_mean, 4),
    }


# ─────────────────────────────────────────────
# Análisis de contornos anómalos
# ─────────────────────────────────────────────

def analyze_contours(canny_edges: np.ndarray) -> dict:
    """
    Encuentra contornos en la imagen de bordes y filtra los sospechosos.

    Lógica:
        - Contornos muy pequeños  → ruido, ignorar
        - Contornos muy grandes   → borde de imagen, ignorar
        - Contornos intermedios   → candidatos a defecto

    Returns:
        n_suspicious    : número de contornos sospechosos
        suspicious_area : área total cubierta por contornos sospechosos
        contours_img    : imagen con contornos dibujados (para visualizar)
        contour_list    : lista de contornos sospechosos (para dibujar)
    """
    total_area = canny_edges.shape[0] * canny_edges.shape[1]
    min_area   = total_area * MIN_CONTOUR_AREA_PCT
    max_area   = total_area * MAX_CONTOUR_AREA_PCT

    # Encontrar todos los contornos
    contours, _ = cv2.findContours(
        canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filtrar contornos sospechosos por área
    suspicious = []
    total_suspicious_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            suspicious.append(cnt)
            total_suspicious_area += area

    # Imagen con contornos dibujados (para debug y visualización)
    contours_img = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contours_img, suspicious, -1, (0, 255, 100), 1)

    return {
        "n_suspicious":     len(suspicious),
        "suspicious_area":  round(total_suspicious_area / total_area, 5),
        "contours_img":     contours_img,
        "contour_list":     suspicious,
    }


# Para detectar defectos en carpet
def compute_texture_diff_score(gray: np.ndarray,
                                baseline_mean_img: np.ndarray) -> float:
    """
    Compara la imagen contra la imagen promedio del baseline.
    Funciona mejor que Canny para texturas orgánicas (carpet, leather).
    """
    # Asegurar mismo tamaño
    if gray.shape != baseline_mean_img.shape:
        baseline_mean_img = cv2.resize(baseline_mean_img, 
                                        (gray.shape[1], gray.shape[0]))
    
    # Diferencia absoluta píxel a píxel
    diff = cv2.absdiff(gray, baseline_mean_img)
    
    # Umbralizar la diferencia — solo nos importan diferencias grandes
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    
    # Score = % de píxeles muy diferentes al baseline
    score = float(np.count_nonzero(thresh)) / (gray.shape[0] * gray.shape[1])
    return round(score, 5)

def analyze_patches(gray: np.ndarray,
                    patch_size: int = 32) -> dict:
    """
    Divide la imagen en parches y detecta cuáles son anómalos
    respecto al comportamiento estadístico del resto.
    
    Un parche es anómalo si su desviación estándar local es muy
    diferente a la mediana de todos los parches de la imagen.
    """
    h, w   = gray.shape
    stds   = []
    means  = []
    coords = []

    # Calcular std y mean de cada parche
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size]
            stds.append(patch.std())
            means.append(patch.mean())
            coords.append((y, x))

    stds  = np.array(stds)
    means = np.array(means)

    # Mediana y MAD (Median Absolute Deviation) — robusto a outliers
    median_std = np.median(stds)
    mad        = np.median(np.abs(stds - median_std))

    # Un parche es anómalo si se desvía más de 3 MADs de la mediana
    threshold  = median_std + 3.0 * (mad + 1e-6)
    anomalous  = stds > threshold

    # Score = fracción de parches anómalos
    patch_score = float(anomalous.sum()) / len(stds)

    # Mapa visual de anomalías
    anomaly_map = np.zeros_like(gray)
    for i, (y, x) in enumerate(coords):
        if anomalous[i]:
            anomaly_map[y:y+patch_size, x:x+patch_size] = 255

    return {
        "patch_score":   round(patch_score, 5),
        "n_patches":     len(stds),
        "n_anomalous":   int(anomalous.sum()),
        "anomaly_map":   anomaly_map,
    }

# ─────────────────────────────────────────────
# Score de defecto
# ─────────────────────────────────────────────

def compute_defect_score(texture: dict,
                          edges: dict,
                          contours: dict,
                          baseline: Optional[dict] = None) -> float:
    """
    Combina todas las métricas en un score único de 0.0 a 1.0.

        0.0 → tela perfecta
        1.0 → defecto severo

    Fórmula ponderada:
        score = 0.30 * edge_component (bordes)
              + 0.35 * contour_component (contornos)
              + 0.20 * texture_component (textura)
              + 0.15 * sobel_component (gradiente)

    Si se provee un baseline (métricas de imágenes normales),
    el score es relativo a ese baseline — mucho más preciso.
    """
    # Componente de bordes: densidad de bordes Canny
    # Valores típicos normales: 0.05 - 0.15
    edge_raw = edges["edge_density"]
    if baseline:
        edge_comp = (edge_raw - baseline["edge_density"]) / (baseline["edge_density"] + 1e-6)
        edge_comp = float(np.clip(edge_comp, 0.0, 1.0))
    else:
        edge_comp = float(np.clip(edge_raw * 4.0, 0.0, 1.0))

    # Componente de contornos: área sospechosa relativa
    contour_comp = float(np.clip(contours["suspicious_area"] * 8.0, 0.0, 1.0))

    # Componente de textura: baja uniformidad = sospechoso
    texture_comp = float(np.clip(1.0 - texture["uniformity"], 0.0, 1.0))

    # Componente Sobel: gradiente promedio alto = bordes abruptos anómalos
    sobel_raw  = edges["sobel_mean"]
    if baseline:
        sobel_comp = (sobel_raw - baseline["sobel_mean"]) / (baseline["sobel_mean"] + 1e-6)
        sobel_comp = float(np.clip(sobel_comp, 0.0, 1.0))
    else:
        sobel_comp = float(np.clip(sobel_raw / 30.0, 0.0, 1.0))

    score = (0.30 * edge_comp
           + 0.35 * contour_comp
           + 0.20 * texture_comp
           + 0.15 * sobel_comp)

    return round(float(np.clip(score, 0.0, 1.0)), 4)


# ─────────────────────────────────────────────
# Analizador de imagen individual
# ─────────────────────────────────────────────

def analyze_image(img_path: Path,
                  baseline: Optional[dict] = None) -> dict:
    """
    Método completo para una sola imagen y para analizar por parches en el caso de carpet.
    Retorna un dict con todas las métricas y las imágenes intermedias.
    """
    imgs      = preprocess(img_path)
    texture   = analyze_texture(imgs["gray"])
    edges     = detect_edges(imgs["blurred"])
    contours  = analyze_contours(edges["canny_edges"])
    patches   = analyze_patches(imgs["gray"])          

    # Score final — estrategia según tipo de tela
    if baseline and baseline.get("edge_density", 0) > 0.30:
        # Tela densa (carpet) → score basado en parches locales
        contour_signal = min(contours["suspicious_area"] * 15.0, 1.0)
        patch_signal   = min(patches["patch_score"] * 5.0, 1.0)
        score = round(max(patch_signal, contour_signal * 0.6), 4)
    elif baseline and "mean_img" in baseline:
        # Tela con baseline de imagen → diferencia global
        texture_diff  = compute_texture_diff_score(imgs["gray"],
                                                    baseline["mean_img"])
        score_original = compute_defect_score(texture, edges, contours, baseline)
        score = round(max(score_original, texture_diff * 1.5), 4)
    else:
        score = compute_defect_score(texture, edges, contours, baseline)

    return {
        # Imágenes para visualización
        "img_original":  imgs["resized"],
        "img_gray":      imgs["gray"],
        "img_edges":     edges["canny_edges"],
        "img_sobel":     edges["sobel_mag"],
        "img_contours":  contours["contours_img"],
        # Métricas numéricas
        "edge_density":  edges["edge_density"],
        "sobel_mean":    edges["sobel_mean"],
        "std_local":     texture["std_local"],
        "uniformity":    texture["uniformity"],
        "entropy":       texture["entropy"],
        "n_suspicious":  contours["n_suspicious"],
        "suspicious_area": contours["suspicious_area"],
        "patch_score":   patches["patch_score"],
        "n_anomalous":   patches["n_anomalous"],
        "defect_score":  score,
        "is_defect":     score >= DEFECT_SCORE_THRESH,
    }


# ─────────────────────────────────────────────
# Construcción del baseline
# ─────────────────────────────────────────────

def build_baseline(good_dir: Path,
                   max_images: int = 30) -> dict:
    """
    Calcula métricas promedio de imágenes SIN defecto.
    Este baseline hace el score mucho más preciso porque
    es relativo a lo que es 'normal' para ESA tela específica.
    """
    print(f"  Construyendo baseline con imágenes de: {good_dir}")
    img_paths = list(good_dir.glob("*.png"))[:max_images]
    if not img_paths:
        img_paths = list(good_dir.glob("*.jpg"))[:max_images]

    if not img_paths:
        print(" No se encontraron imágenes para baseline. Usando valores absolutos.")
        return {}

    edge_densities, sobel_means = [], []

    for p in img_paths:
        try:
            imgs  = preprocess(p)
            edges = detect_edges(imgs["blurred"])
            edge_densities.append(edges["edge_density"])
            sobel_means.append(edges["sobel_mean"])
        except Exception:
            continue

    baseline = {
        "edge_density": float(np.mean(edge_densities)),
        "sobel_mean":   float(np.mean(sobel_means)),
        "n_images":     len(img_paths),
    }
    print(f" Baseline calculado con {baseline['n_images']} imágenes")
    print(f"  edge_density media : {baseline['edge_density']:.5f}")
    print(f"  sobel_mean media   : {baseline['sobel_mean']:.3f}")
    
    # Calcular imagen promedio del baseline
    mean_imgs = []
    for p in img_paths:
        try:
            img   = cv2.imread(str(p))
            img   = cv2.resize(img, IMG_SIZE)
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_imgs.append(gray.astype(np.float32))
        except Exception:
            continue

    baseline_mean_img = np.mean(mean_imgs, axis=0).astype(np.uint8)

    baseline = {
        "edge_density":     float(np.mean(edge_densities)),
        "sobel_mean":       float(np.mean(sobel_means)),
        "n_images":         len(img_paths),
        "mean_img":         baseline_mean_img,   # ← NUEVO
    }
    print(f"  Baseline calculado con {baseline['n_images']} imágenes")
    print(f"    edge_density media : {baseline['edge_density']:.5f}")
    print(f"    sobel_mean media   : {baseline['sobel_mean']:.3f}")
    return baseline


# ─────────────────────────────────────────────
# Análisis por lotes (MVTec)
# ─────────────────────────────────────────────

def analyze_mvtec_category(category_path: Path,
                            output_dir: Path,
                            max_per_class: int = 15) -> pd.DataFrame:
    """
    Procesa una categoría completa de MVTec AD que es nuestra base de datos
    Con dos categorias de ejemplo: "carpet" y "grid".

    Retorna DataFrame con métricas de todas las imágenes procesadas.
    """
    print(f"\n{'═'*58}")
    print(f"  Analizando categoría: {category_path.name.upper()}")
    print(f"{'═'*58}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline con imágenes de entrenamiento normales
    train_good = category_path / "train" / "good"
    baseline = build_baseline(train_good)

        # Umbral adaptativo según complejidad de la tela
        # Carpet es muy densa, necesita umbral más bajo
    if baseline.get("edge_density", 0) > 0.30:
        adaptive_threshold = 0.18
        print(f" Tela densa detectada => umbral adaptativo: {adaptive_threshold}")
    else:
        adaptive_threshold = DEFECT_SCORE_THRESH  # 0.35 para grid
        print(f"  Umbral estándar: {adaptive_threshold}")

    # 2. Recorrer todas las subcarpetas de test/
    test_dir = category_path / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"No existe: {test_dir}")

    records = []
    defect_classes = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    for defect_class in defect_classes:
        label       = defect_class.name          # "good", "hole", "cut", etc.
        is_defect_gt = label != "good"           # ground truth: ¿tiene defecto?

        img_paths = sorted(list(defect_class.glob("*.png")) +
                           list(defect_class.glob("*.jpg")))[:max_per_class]

        print(f"\n  Clase: {label:25s} ({len(img_paths)} imágenes)")

        for img_path in img_paths:
            try:
                result = analyze_image(img_path, baseline=baseline)
                records.append({
                    "category":        category_path.name,
                    "defect_class":    label,
                    "is_defect_gt":    is_defect_gt,
                    "is_defect_pred":  result["defect_score"] >= adaptive_threshold,
                    "defect_score":    result["defect_score"],
                    "edge_density":    result["edge_density"],
                    "sobel_mean":      result["sobel_mean"],
                    "uniformity":      result["uniformity"],
                    "entropy":         result["entropy"],
                    "n_suspicious":    result["n_suspicious"],
                    "suspicious_area": result["suspicious_area"],
                    "img_path":        str(img_path),
                })
            except Exception as e:
                print(f"   Error en {img_path.name}: {e}")

    df = pd.DataFrame(records)

    # 3. Métricas de clasificación
    _print_classification_metrics(df, category_path.name)

    # 4. Guardar resultados
    csv_path = output_dir / f"vision_results_{category_path.name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Resultados guardados: {csv_path}")

    # 5. Visualizaciones
    _plot_score_distribution(df, category_path.name, output_dir)
    _plot_sample_analysis(df, category_path, output_dir)

    return df


# ─────────────────────────────────────────────
# Métricas de clasificación
# ─────────────────────────────────────────────

def _print_classification_metrics(df: pd.DataFrame, category: str) -> None:
    """Imprime precisión, recall y F1 del detector."""
    tp = ((df["is_defect_gt"] == True)  & (df["is_defect_pred"] == True)).sum()
    tn = ((df["is_defect_gt"] == False) & (df["is_defect_pred"] == False)).sum()
    fp = ((df["is_defect_gt"] == False) & (df["is_defect_pred"] == True)).sum()
    fn = ((df["is_defect_gt"] == True)  & (df["is_defect_pred"] == False)).sum()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy  = (tp + tn) / len(df)

    print(f"\n  {'─'*40}")
    print(f"  MÉTRICAS — {category.upper()}")
    print(f"  {'─'*40}")
    print(f"  Accuracy  : {accuracy:.1%}")
    print(f"  Precision : {precision:.1%}")
    print(f"  Recall    : {recall:.1%}")
    print(f"  F1 Score  : {f1:.1%}")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")


# ─────────────────────────────────────────────
# Visualizaciones
# ─────────────────────────────────────────────

def _plot_score_distribution(df: pd.DataFrame,
                              category: str,
                              output_dir: Path) -> None:
    """Histograma de scores: normal vs defecto."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    normal_scores  = df[df["is_defect_gt"] == False]["defect_score"]
    defect_scores  = df[df["is_defect_gt"] == True]["defect_score"]

    bins = np.linspace(0, 1, 30)
    ax.hist(normal_scores, bins=bins, color="#00d4ff",
            alpha=0.7, label=f"Normal (n={len(normal_scores)})")
    ax.hist(defect_scores, bins=bins, color="#ff4757",
            alpha=0.7, label=f"Defecto (n={len(defect_scores)})")
    ax.axvline(DEFECT_SCORE_THRESH, color="#ffa502",
               linewidth=2, linestyle="--",
               label=f"Umbral ({DEFECT_SCORE_THRESH})")

    ax.set_title(f"Distribución de Scores — {category.upper()}",
                 color="white", fontsize=12)
    ax.set_xlabel("Defect Score", color="#aaaaaa")
    ax.set_ylabel("Frecuencia", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_color("#333355")
    ax.grid(True, color="#222244", linewidth=0.4, alpha=0.7)

    out = output_dir / f"score_distribution_{category}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Figura: {out}")


def _plot_sample_analysis(df: pd.DataFrame,
                           category_path: Path,
                           output_dir: Path,
                           n_samples: int = 3) -> None:
    """
    Muestra el método visual completo para N imágenes de ejemplo:
    original → grises → bordes Canny → Sobel → contornos
    """
    # Tomar muestras: 1-2 normales + resto defectuosas
    good_sample    = df[df["is_defect_gt"] == False].head(1)
    defect_sample  = df[df["is_defect_gt"] == True].head(n_samples - 1)
    samples        = pd.concat([good_sample, defect_sample])

    n_rows = len(samples)
    fig    = plt.figure(figsize=(18, 4 * n_rows), facecolor="#0f0f1a")
    fig.suptitle(f"Método de Análisis Visual — {category_path.name.upper()}",
                 color="white", fontsize=13, fontweight="bold")

    col_titles = ["Original", "Escala de grises",
                  "Bordes (Canny)", "Gradiente (Sobel)", "Contornos"]

    baseline = build_baseline(category_path / "train" / "good")

    for row_idx, (_, record) in enumerate(samples.iterrows()):
        result = analyze_image(Path(record["img_path"]), baseline=baseline)
        imgs_to_show = [
            result["img_original"],
            result["img_gray"],
            result["img_edges"],
            result["img_sobel"],
            result["img_contours"],
        ]
        label_str = (f"[{record['defect_class'].upper()}] "
                     f"Score: {result['defect_score']:.3f} | "
                     f"{'DEFECTO' if result['is_defect'] else 'NORMAL'}")

        for col_idx, (img, col_title) in enumerate(zip(imgs_to_show, col_titles)):
            ax = fig.add_subplot(n_rows, 5, row_idx * 5 + col_idx + 1)
            ax.set_facecolor("#1a1a2e")

            # OpenCV usa BGR, matplotlib usa RGB
            if len(img.shape) == 3:
                display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(display)
            else:
                ax.imshow(img, cmap="gray")

            if row_idx == 0:
                ax.set_title(col_title, color="#aaaaaa", fontsize=8, pad=4)
            if col_idx == 0:
                ax.set_ylabel(label_str, color="white" if record["is_defect_gt"]
                              else "#00d4ff", fontsize=7, labelpad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#ff4757" if record["is_defect_gt"] else "#00d4ff")
                sp.set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = output_dir / f"sample_{category_path.name}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Figura del método: {out}")


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Ruta a la carpeta raíz con las imagenes MVTec AD 
    MVTEC_ROOT = Path("Datos_Imagenes_Tela")
    OUTPUT_DIR = Path("data/vision_output")

    categories = ["carpet", "grid"]

    all_results = []
    for cat in categories:
        cat_path = MVTEC_ROOT / cat
        if not cat_path.exists():
            print(f"  No encontrado: {cat_path}  — omitiendo")
            continue
        df = analyze_mvtec_category(cat_path, OUTPUT_DIR)
        all_results.append(df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "vision_results_all.csv", index=False)
        print(f"\n  Resultado combinado: {OUTPUT_DIR / 'vision_results_all.csv'}")