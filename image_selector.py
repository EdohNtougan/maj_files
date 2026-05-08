# image_selector.py
"""
Sélection d'images pour articles WordPress — v3.0 CLIP-Enhanced
================================================================

Nouveauté v3.0 : Intégration CLIP (openai/clip-vit-base-patch32)
  - Remplace le score φ_relevance (Jaccard URL) par φ_clip (0.0–1.0)
  - Chargement LAZY : CLIP n'est chargé qu'au moment du scoring final
  - Libération immédiate après usage (del model → RAM restituée)
  - Fallback automatique vers Jaccard si CLIP indisponible

Architecture pipeline (inchangée v2.0) :
  T0  Collecte parallèle    8 threads → OG/Twitter de toutes les sources
  T1  HEAD pré-filtre       12 threads → élimine ~50% sans download
  T2  Scoring métadonnées   Score(C) = Σ wᵢ·φᵢ(C) sans CLIP (pré-download)
  T3  Download top-N        6 threads → validation perceptuelle
  T4  CLIP re-scoring       φ_clip remplace φ_relevance sur top candidats
  T5  pHash dédup           → meilleur score final normalisé
  T6  Fallback SearXNG      même pipeline avec CLIP

Formules mathématiques :
──────────────────────────────────────────────────────────────────────
  Score global  :  S(C) = Σᵢ wᵢ · φᵢ(C)          Σwᵢ = 1.0

  CLIP          :  φ_clip = cosine_sim(E_text, E_image)
                   E_text  = CLIP.encode_text(topic_title)
                   E_image = CLIP.encode_image(candidate_image)
                   → Remplace φ_J (Jaccard URL) en T4

  Aspect ratio  :  φ_r = exp( -((r − r*) / σ)² )   r* = 16/9, σ = 0.4
  Dimensions    :  φ_d = clamp( (P − Pmin) / (Pmax − Pmin), 0, 1 )
  Entropie      :  H = −Σ pₖ log₂(pₖ)             (Shannon, grayscale)
  Netteté       :  σ²L = Var(∇²I)                  (Laplacien 3×3)
  pHash dédup   :  dH(a,b) = popcount(a ⊕ b)       seuil < 12 → doublon
  Circuit break :  T_reset = T_base × 2^(fails − N)  (backoff exponentiel)

Dépendances CLIP (optionnelles) :
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install transformers Pillow

Dépendances pipeline (requises) :
  pip install requests beautifulsoup4 lxml pillow numpy
════════──────────────────────────────────────────────────────────────
"""

import gc
import hashlib
import io
import logging
import math
import os
import threading
import time
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageFilter

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ── Détection CLIP (optionnel) ──────────────────────────────────────
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

logger = logging.getLogger(__name__)

# =====================================================================
# CONSTANTES — Dimensions et validation
# =====================================================================
MIN_WIDTH = 800
MIN_HEIGHT = 400
TARGET_MAX_WIDTH = 1920
MAX_FILE_SIZE_MB = 8
JPEG_QUALITY = 85

# =====================================================================
# CONSTANTES — Aspect Ratio (modèle Gaussien)
# =====================================================================
OPTIMAL_RATIO = 16 / 9
RATIO_SIGMA = 0.4
MIN_ASPECT_RATIO = 1.2
MAX_ASPECT_RATIO = 4.0

# =====================================================================
# CONSTANTES — Pondérations scoring multi-critères v3.0
#
#   CHANGEMENT v3.0 :
#   'relevance' passe de 0.18 → 0.08  (Jaccard URL, fallback uniquement)
#   'clip'      ajouté à  0.25        (score sémantique CLIP)
#   Les autres critères restent inchangés.
#   Somme : 0.10 + 0.15 + 0.14 + 0.20 + 0.13 + 0.08 + 0.20 = 1.00 ✓
#
#   Quand CLIP est indisponible :
#   'relevance' reprend 0.20, les autres sont renormalisés à Σ=1.0
# =====================================================================
WEIGHTS_WITH_CLIP = {
    'source':     0.10,   # OG (1.0) > Twitter (0.75) > SearXNG (0.50)
    'dimensions': 0.15,   # Surface en pixels (normalisée min-max)
    'aspect':     0.14,   # Proximité au ratio 16/9 (Gaussienne)
    'entropy':    0.20,   # Richesse informationnelle (Shannon)
    'sharpness':  0.13,   # Netteté (variance Laplacienne)
    'relevance':  0.08,   # Jaccard URL (fallback/appoint)
    'clip':       0.20,   # Pertinence sémantique CLIP ← NOUVEAU
}
# Vérif : 0.10+0.15+0.14+0.20+0.13+0.08+0.20 = 1.00 ✓

WEIGHTS_WITHOUT_CLIP = {
    'source':     0.10,
    'dimensions': 0.18,
    'aspect':     0.17,
    'entropy':    0.22,
    'sharpness':  0.15,
    'relevance':  0.18,
}
# Vérif : 0.10+0.18+0.17+0.22+0.15+0.18 = 1.00 ✓

SOURCE_SCORES = {
    'opengraph':    1.00,
    'twitter_card': 0.75,
    'searxng':      0.50,
}

# =====================================================================
# CONSTANTES — Seuils perceptuels
# =====================================================================
MIN_ENTROPY = 3.0
MAX_ENTROPY = 7.5
MIN_LAPLACIAN_VAR = 40.0
MAX_LAPLACIAN_VAR = 500.0
PHASH_THRESHOLD = 12

# =====================================================================
# CONSTANTES — Pixels
# =====================================================================
MIN_PIXELS = MIN_WIDTH * MIN_HEIGHT
MAX_PIXELS = TARGET_MAX_WIDTH * int(TARGET_MAX_WIDTH * 9 / 16)

# =====================================================================
# CONSTANTES — Parallélisme
# =====================================================================
MAX_SOURCE_WORKERS = 8
MAX_HEAD_WORKERS = 12
MAX_DOWNLOAD_WORKERS = 6
MAX_SEARXNG_WORKERS = 3
TOP_N_DOWNLOAD = 4
MAX_CANDIDATE_BATCHES = 3
HEAD_TIMEOUT = 5
DOWNLOAD_TIMEOUT = 20
SOURCE_FETCH_TIMEOUT = 10
SEARXNG_TIMEOUT = 15
URL_CACHE_MAXSIZE = 512

# =====================================================================
# CLIP — Configuration
# =====================================================================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_MAX_CANDIDATES = 6      # Nombre max de candidats scorés par CLIP
CLIP_IMAGE_SIZE = (224, 224) # Taille attendue par CLIP
CLIP_MIN_SCORE = 0.15        # Score cosine minimal pour considérer l'image

# =====================================================================
# CONTEXTES SearXNG
# =====================================================================
CATEGORY_CONTEXT = {
    "AI_TECHNOLOGY": "artificial intelligence technology",
    "CRYPTO":        "cryptocurrency blockchain",
    "FINANCE":       "financial markets economy",
    "GEOPOLITICS":   "geopolitics world politics",
    "SCIENCE":       "scientific research discovery",
    "HEALTH":        "health medical research",
    "GENERAL":       "news latest",
}

URL_REJECT_PATTERNS = frozenset({
    'thumb', 'thumbnail', 'icon', 'logo', 'avatar', 'sprite',
    'favicon', 'emoji', 'badge', 'button', 'pixel', 'spacer',
    'blank', 'loading', 'placeholder', '1x1', 'tracking',
    'flags', 'spinner',
})


# =====================================================================
# Exceptions
# =====================================================================
class ImageValidationError(Exception):
    pass


# =====================================================================
# CLIP Engine — Chargement lazy + libération mémoire
# =====================================================================
class CLIPEngine:
    """
    Moteur CLIP avec chargement LAZY et libération mémoire explicite.

    Principe :
      - Le modèle n'est JAMAIS chargé au démarrage du daemon.
      - Il est chargé UNIQUEMENT dans score_candidates(), puis libéré.
      - RAM consommée : ~1.5Go pendant max 30s par cycle d'images.
      - Thread-safe via verrou interne.

    Usage :
        clip = CLIPEngine()
        scores = clip.score_candidates(topic, [img1, img2, img3])
        # Le modèle est automatiquement libéré après cette ligne.
    """

    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        self.model_name = model_name
        self._lock = threading.Lock()
        self._available: Optional[bool] = None  # None = pas encore testé

    def is_available(self) -> bool:
        """Vérifie si CLIP peut être chargé (torch + transformers installés)."""
        if self._available is not None:
            return self._available
        if not HAS_CLIP:
            self._available = False
            logger.warning(
                "⚠️ CLIP indisponible — torch/transformers non installés. "
                "Fallback Jaccard actif."
            )
        else:
            self._available = True
            logger.info("✅ CLIP disponible — modèle : %s", self.model_name)
        return self._available

    def score_candidates(
        self,
        topic: str,
        candidate_images: List[Tuple[str, Image.Image]],
    ) -> Dict[str, float]:
        """
        Score une liste de candidats (url, PIL.Image) contre un topic texte.

        Algorithme :
          1. Encoder le topic en vecteur texte E_text (1 × 512)
          2. Encoder chaque image en vecteur visuel E_img (N × 512)
          3. Calculer cosine_similarity(E_text, E_img_i) pour chaque image
          4. Normaliser dans [0, 1] par rapport au max du lot
          5. Libérer le modèle immédiatement

        Retourne :
          {url: clip_score_normalisé, ...}
          clip_score ∈ [0.0, 1.0]
        """
        if not self.is_available() or not candidate_images:
            return {url: 0.0 for url, _ in candidate_images}

        scores: Dict[str, float] = {}

        with self._lock:
            model = None
            processor = None
            try:
                logger.info(
                    "🤖 CLIP chargement pour %d images — %s",
                    len(candidate_images), topic[:50],
                )
                t_load = time.time()

                # ── Chargement du modèle ──────────────────────────────
                model = CLIPModel.from_pretrained(self.model_name)
                processor = CLIPProcessor.from_pretrained(self.model_name)
                model.eval()

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)

                logger.info(
                    "🤖 CLIP chargé en %.1fs sur %s",
                    time.time() - t_load, device.upper(),
                )

                # ── Encodage du texte ─────────────────────────────────
                with torch.no_grad():
                    text_inputs = processor(
                        text=[topic],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77,
                    ).to(device)
                    text_features = model.get_text_features(**text_inputs)
                    # Normalisation L2
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )

                # ── Encodage des images ───────────────────────────────
                raw_scores: List[Tuple[str, float]] = []

                for url, pil_img in candidate_images:
                    try:
                        # Redimensionner pour CLIP (224×224)
                        img_resized = pil_img.convert("RGB").resize(
                            CLIP_IMAGE_SIZE, Image.Resampling.LANCZOS
                        )
                        with torch.no_grad():
                            img_inputs = processor(
                                images=img_resized,
                                return_tensors="pt",
                            ).to(device)
                            img_features = model.get_image_features(**img_inputs)
                            img_features = img_features / img_features.norm(
                                dim=-1, keepdim=True
                            )

                        # cosine similarity = dot product (vecteurs normalisés L2)
                        cosine_sim = float(
                            (text_features * img_features).sum(dim=-1).item()
                        )
                        # CLIP retourne des valeurs entre -1 et 1 en théorie,
                        # mais en pratique images/texte réels sont entre 0.1 et 0.4.
                        # On clippe à [0, 1]
                        cosine_sim = max(0.0, cosine_sim)
                        raw_scores.append((url, cosine_sim))

                        logger.debug(
                            "  🤖 CLIP [%.3f] — %s",
                            cosine_sim, url[-45:],
                        )
                    except Exception as img_err:
                        logger.debug(
                            "  ⚠️ CLIP erreur image [%s]: %s",
                            url[-30:], str(img_err)[:60],
                        )
                        raw_scores.append((url, 0.0))

                # ── Normalisation dans [0, 1] par rapport au max du lot ──
                if raw_scores:
                    max_sim = max(s for _, s in raw_scores)
                    min_sim = min(s for _, s in raw_scores)
                    spread = max_sim - min_sim if max_sim > min_sim else 1.0

                    for url, sim in raw_scores:
                        # Normalisation min-max dans le lot
                        norm = (sim - min_sim) / spread if spread > 0 else 0.5
                        scores[url] = round(norm, 4)

                        logger.info(
                            "  🎯 CLIP score normalisé %.3f (raw=%.3f) — %s",
                            norm, sim, url[-40:],
                        )

            except Exception as e:
                logger.error("❌ CLIP scoring error: %s", e)
                scores = {url: 0.0 for url, _ in candidate_images}

            finally:
                # ── LIBÉRATION MÉMOIRE IMMÉDIATE ─────────────────────
                if model is not None:
                    del model
                if processor is not None:
                    del processor
                gc.collect()
                if HAS_CLIP and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("♻️ CLIP libéré de la mémoire.")

        return scores


# =====================================================================
# Circuit Breaker
# =====================================================================
class _CircuitBreaker:
    def __init__(self, max_failures: int = 3, base_reset: float = 60.0):
        self.max_failures = max_failures
        self.base_reset = base_reset
        self._failures: Dict[str, int] = defaultdict(int)
        self._last_fail: Dict[str, float] = {}
        self._lock = threading.Lock()

    def is_healthy(self, domain: str) -> bool:
        with self._lock:
            failures = self._failures.get(domain, 0)
            if failures < self.max_failures:
                return True
            elapsed = time.time() - self._last_fail.get(domain, 0)
            reset_time = self.base_reset * (2 ** (failures - self.max_failures))
            if elapsed > reset_time:
                self._failures[domain] = 0
                return True
            return False

    def record_failure(self, domain: str) -> None:
        with self._lock:
            self._failures[domain] += 1
            self._last_fail[domain] = time.time()

    def record_success(self, domain: str) -> None:
        with self._lock:
            if domain in self._failures:
                self._failures[domain] = 0


# =====================================================================
# Cache LRU HEAD
# =====================================================================
class _HeadCache:
    def __init__(self, maxsize: int = URL_CACHE_MAXSIZE):
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: str, value: Dict) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)


# =====================================================================
# SELECTEUR D'IMAGES v3.0 — Pipeline CLIP-Enhanced
# =====================================================================
class ImageSelector:
    """
    Sélection d'images WordPress — Pipeline parallèle + CLIP v3.0

    Workflow complet :
      1. Extraction parallèle OG/Twitter (8 threads)
      2. HEAD pré-filtre parallèle (12 threads → ~50% éliminés)
      3. Scoring métadonnées sans CLIP (5 critères, pré-download)
      4. Download parallèle top-N (6 threads) + validation perceptuelle
      5. CLIP re-scoring des candidats valides (φ_clip remplace φ_relevance)
      6. pHash déduplication + sélection finale
      7. Fallback SearXNG avec même pipeline CLIP si besoin
    """

    OUTPUT_DIR = "./assets/images/featured"

    def __init__(self, searxng_url: str = "http://127.0.0.1:8888"):
        self.searxng_url = searxng_url.rstrip("/")
        self._circuit = _CircuitBreaker()
        self._head_cache = _HeadCache()
        self._thread_local = threading.local()
        self._clip = CLIPEngine()
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        logger.info(
            "✅ ImageSelector v3.0 initialisé — CLIP=%s | %s",
            "ON" if self._clip.is_available() else "OFF (fallback Jaccard)",
            self.OUTPUT_DIR,
        )

    # =================================================================
    # Thread-local Session
    # =================================================================
    def _get_session(self) -> requests.Session:
        if not hasattr(self._thread_local, 'session') or \
                self._thread_local.session is None:
            session = requests.Session()
            session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "TrendMonitor/3.0 (compatible; article-pipeline)"
                )
            })
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=10,
                max_retries=2,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._thread_local.session = session
        return self._thread_local.session

    # =================================================================
    # Point d'entrée public
    # =================================================================
    def select_featured_image(
        self,
        topic: str,
        source_items: List[Dict[str, Any]],
        category: str = "GENERAL",
    ) -> Optional[Dict[str, Any]]:
        """
        Retourne le meilleur dict image pour le topic, ou None.

        Retour :
          {
            'local_path':    str,
            'original_url':  str,
            'width':         int,
            'height':        int,
            'file_size_kb':  int,
            'filename':      str,
            'source':        'opengraph' | 'twitter_card' | 'searxng',
            'score':         float,
            'clip_score':    float,  ← NOUVEAU v3.0
            'clip_used':     bool,   ← NOUVEAU v3.0
          }
        """
        logger.info("🖼️ Recherche image CLIP — %s [%s]", topic[:70], category)
        t_start = time.time()

        # ── STAGE 1 : Collecte parallèle OG/Twitter ──
        candidates = self._extract_all_candidates_parallel(source_items)
        candidates = self._deduplicate_urls(candidates)
        logger.info(
            "  T1: %d candidats OG/Twitter uniques (%.1fs)",
            len(candidates), time.time() - t_start,
        )

        if candidates:
            result = self._pipeline(candidates, topic)
            if result:
                elapsed = time.time() - t_start
                logger.info(
                    "✅ Image sélectionnée (%.1fs) — %dx%d | score=%.3f "
                    "| clip=%.3f | %s",
                    elapsed, result['width'], result['height'],
                    result.get('score', 0),
                    result.get('clip_score', 0),
                    result['source'],
                )
                return result

        # ── STAGE 6 : Fallback SearXNG ──
        logger.info("⚠️ Fallback SearXNG — %s [%s]", topic[:60], category)
        t_searx = time.time()
        result = self._searxng_fallback(topic, category)
        elapsed = time.time() - t_start

        if result:
            logger.info(
                "✅ Image SearXNG (%.1fs) — %dx%d | score=%.3f | clip=%.3f",
                elapsed, result['width'], result['height'],
                result.get('score', 0), result.get('clip_score', 0),
            )
        else:
            logger.warning("❌ Aucune image valide (%.1fs)", elapsed)

        return result

    # =================================================================
    # Pipeline interne
    # =================================================================
    def _pipeline(
        self, candidates: List[Dict], topic: str
    ) -> Optional[Dict]:
        """
        T1 HEAD → T2 Score métadonnées → T3 Download → T4 CLIP → T5 pHash
        """
        # T1 : HEAD pré-filtre
        viable = self._head_prefilter_parallel(candidates)
        if not viable:
            return None

        # T2 : Scoring métadonnées (sans CLIP — pré-download)
        scored = self._score_by_metadata(viable, topic)

        # T3 + T4 + T5 : Download adaptatif par batch
        offset = 0
        for batch_idx in range(MAX_CANDIDATE_BATCHES):
            n = self._adaptive_top_n(scored, offset)
            batch = scored[offset: offset + n]
            if not batch:
                break

            logger.debug(
                "  T3 batch %d: %d candidats (offset=%d)",
                batch_idx + 1, len(batch), offset,
            )

            # T3 : Download + validation perceptuelle
            validated = self._download_and_validate_parallel(batch, topic)

            if validated:
                # T4 : CLIP re-scoring (remplace φ_relevance)
                validated = self._apply_clip_rescoring(validated, topic)

                # T5 : pHash déduplication
                validated = self._phash_dedup(validated)
                best = max(validated, key=lambda r: r.get('_final_score', 0))
                return self._clean_result(best)

            offset += n

        return None

    # =================================================================
    # T4 — CLIP Re-scoring (NOUVEAU v3.0)
    # =================================================================
    def _apply_clip_rescoring(
        self,
        validated: List[Dict],
        topic: str,
    ) -> List[Dict]:
        """
        Applique CLIP sur les candidats validés et recalcule le score final.

        Stratégie :
          1. Recharger les images PIL depuis le fichier local sauvegardé
          2. Passer au CLIPEngine (max CLIP_MAX_CANDIDATES candidats)
          3. Intégrer φ_clip dans le score final pondéré
          4. Libérer les images PIL immédiatement après

        Si CLIP indisponible → score inchangé (Jaccard URL conservé).
        """
        if not self._clip.is_available() or not validated:
            logger.debug(
                "  T4: CLIP absent — %d candidats conservent score Jaccard",
                len(validated),
            )
            return validated

        # Limiter le nombre de candidats CLIP pour économiser la RAM
        top_for_clip = validated[:CLIP_MAX_CANDIDATES]
        rest = validated[CLIP_MAX_CANDIDATES:]

        # Charger les images PIL depuis le fichier local
        clip_inputs: List[Tuple[str, Image.Image]] = []
        for r in top_for_clip:
            local_path = r.get('local_path', '')
            url = r.get('original_url', '')
            try:
                if local_path and os.path.exists(local_path):
                    pil_img = Image.open(local_path).convert("RGB")
                    clip_inputs.append((url, pil_img))
                else:
                    clip_inputs.append((url, None))
            except Exception as e:
                logger.debug("  ⚠️ Lecture image locale [%s]: %s", local_path, e)
                clip_inputs.append((url, None))

        # Filtrer les images qui ont pu être chargées
        valid_clip_inputs = [(url, img) for url, img in clip_inputs if img is not None]
        failed_urls = {url for url, img in clip_inputs if img is None}

        # Score CLIP
        clip_scores: Dict[str, float] = {}
        if valid_clip_inputs:
            clip_scores = self._clip.score_candidates(topic, valid_clip_inputs)

        # Libérer les PIL images
        for _, img in valid_clip_inputs:
            if img is not None:
                img.close()
        del valid_clip_inputs
        gc.collect()

        # Recalculer le score final avec φ_clip
        weights = WEIGHTS_WITH_CLIP
        recalibrated = []
        for r in top_for_clip:
            url = r.get('original_url', '')
            phi_clip = clip_scores.get(url, 0.0)
            r['_clip_score'] = phi_clip

            if url in failed_urls:
                # Image non chargeable → recalcul sans CLIP
                r['_clip_used'] = False
                recalibrated.append(r)
                continue

            # Recalculer avec le poids CLIP
            phi_source   = SOURCE_SCORES.get(r.get('source', ''), 0.5)
            phi_dim      = self._clamp_normalize(
                (r.get('width', 0) * r.get('height', 0)), MIN_PIXELS, MAX_PIXELS
            )
            phi_aspect   = self._gaussian_aspect(
                r.get('width', 1) / max(r.get('height', 1), 1)
            )
            phi_entropy  = self._clamp_normalize(
                r.get('_entropy', 4.0), MIN_ENTROPY, MAX_ENTROPY
            )
            phi_sharp    = self._clamp_normalize(
                r.get('_sharpness', 100.0), MIN_LAPLACIAN_VAR, MAX_LAPLACIAN_VAR
            )
            phi_rel      = r.get('_phi_relevance', 0.0)  # Jaccard URL (appoint)

            new_score = (
                weights['source']     * phi_source
                + weights['dimensions'] * phi_dim
                + weights['aspect']     * phi_aspect
                + weights['entropy']    * phi_entropy
                + weights['sharpness']  * phi_sharp
                + weights['relevance']  * phi_rel
                + weights['clip']       * phi_clip
            )

            old_score = r.get('_final_score', 0.0)
            r['_final_score'] = new_score
            r['_clip_used'] = True

            logger.info(
                "  🎯 CLIP re-score : %.3f → %.3f (clip=%.3f) | %s",
                old_score, new_score, phi_clip, url[-40:],
            )
            recalibrated.append(r)

        # Marquer les candidats hors top CLIP comme non-scorés CLIP
        for r in rest:
            r['_clip_score'] = 0.0
            r['_clip_used'] = False

        return recalibrated + rest

    # =================================================================
    # STAGE 1 : Extraction parallèle OG / Twitter
    # =================================================================
    def _extract_all_candidates_parallel(
        self, source_items: List[Dict]
    ) -> List[Dict]:
        candidates: List[Dict] = []
        valid_sources = []

        for item in source_items[:12]:
            url = item.get('url')
            if not url:
                continue
            domain = urlparse(url).netloc
            if not self._circuit.is_healthy(domain):
                continue
            valid_sources.append(url)

        if not valid_sources:
            return candidates

        with ThreadPoolExecutor(max_workers=MAX_SOURCE_WORKERS) as pool:
            futures = {
                pool.submit(self._extract_og_from_source, url): url
                for url in valid_sources
            }
            for future in as_completed(futures, timeout=SOURCE_FETCH_TIMEOUT + 5):
                source_url = futures[future]
                try:
                    result = future.result()
                    candidates.extend(result)
                except Exception as e:
                    logger.debug(
                        "  ⚠️ Source échouée %s : %s",
                        source_url[:50], str(e)[:60],
                    )
                    domain = urlparse(source_url).netloc
                    self._circuit.record_failure(domain)

        return candidates

    def _extract_og_from_source(self, page_url: str) -> List[Dict]:
        session = self._get_session()
        candidates: List[Dict] = []

        try:
            resp = session.get(page_url, timeout=SOURCE_FETCH_TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            logger.debug("  ⚠️ Inaccessible %s : %s", page_url[:50], e)
            raise

        soup = BeautifulSoup(resp.text, "lxml")

        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            img_url = self._resolve_url(og_img["content"], page_url)
            w = self._meta_int(soup, "og:image:width")
            h = self._meta_int(soup, "og:image:height")
            candidates.append({
                "url": img_url,
                "source": "opengraph",
                "source_page": page_url,
                "declared_width": w,
                "declared_height": h,
                "content_type": "",
                "content_length": 0,
            })

        if not og_img:
            tw_img = (
                soup.find("meta", attrs={"name": "twitter:image"})
                or soup.find("meta", attrs={"name": "twitter:image:src"})
            )
            if tw_img and tw_img.get("content"):
                candidates.append({
                    "url": self._resolve_url(tw_img["content"], page_url),
                    "source": "twitter_card",
                    "source_page": page_url,
                    "declared_width": 0,
                    "declared_height": 0,
                    "content_type": "",
                    "content_length": 0,
                })

        self._circuit.record_success(urlparse(page_url).netloc)
        return candidates

    # =================================================================
    # STAGE 2 : HEAD pré-filtre parallèle
    # =================================================================
    def _head_prefilter_parallel(
        self, candidates: List[Dict]
    ) -> List[Dict]:
        if not candidates:
            return candidates

        pre_filtered = []
        for c in candidates:
            url_lower = c['url'].lower()
            if any(pat in url_lower for pat in URL_REJECT_PATTERNS):
                continue
            _path = urlparse(c['url']).path.lower()
            if _path.endswith('.svg'):
                continue
            pre_filtered.append(c)

        if not pre_filtered:
            return []

        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        viable: List[Dict] = []

        with ThreadPoolExecutor(max_workers=MAX_HEAD_WORKERS) as pool:
            future_map = {
                pool.submit(self._head_check, c, max_bytes): c
                for c in pre_filtered
            }
            for future in as_completed(future_map, timeout=HEAD_TIMEOUT + 3):
                try:
                    result = future.result()
                    if result is not None:
                        viable.append(result)
                except Exception:
                    c = future_map[future]
                    viable.append(c)

        return viable

    def _head_check(
        self, candidate: Dict, max_bytes: int
    ) -> Optional[Dict]:
        url = candidate['url']
        domain = urlparse(url).netloc

        if not self._circuit.is_healthy(domain):
            return None

        cached = self._head_cache.get(url)
        if cached is not None:
            if cached.get('viable'):
                candidate['content_type'] = cached.get('content_type', '')
                candidate['content_length'] = cached.get('content_length', 0)
                return candidate
            return None

        session = self._get_session()
        try:
            resp = session.head(url, timeout=HEAD_TIMEOUT, allow_redirects=True)
            ct = resp.headers.get('Content-Type', '').lower()
            cl_str = resp.headers.get('Content-Length', '0')
            cl = int(cl_str) if cl_str.isdigit() else 0

            if ct and not ct.startswith('image/'):
                self._head_cache.set(url, {'viable': False})
                return None
            if cl > 0 and cl > max_bytes:
                self._head_cache.set(url, {'viable': False})
                return None

            candidate['content_type'] = ct
            candidate['content_length'] = cl
            self._head_cache.set(url, {
                'viable': True, 'content_type': ct, 'content_length': cl,
            })
            return candidate

        except Exception:
            return candidate

    # =================================================================
    # STAGE 3 : Scoring par métadonnées (sans CLIP — pré-download)
    # =================================================================
    def _score_by_metadata(
        self, candidates: List[Dict], topic: str
    ) -> List[Dict]:
        """
        Score pré-download avec 4 critères (CLIP pas encore disponible).
        Poids renormalisés sans 'clip' et sans 'entropy'/'sharpness'.
        """
        pre_weights = {
            'source': 0.20,
            'dimensions': 0.30,
            'aspect': 0.28,
            'relevance': 0.22,
        }
        topic_keywords = self._extract_keywords(topic)

        for c in candidates:
            phi_source = SOURCE_SCORES.get(c.get('source', ''), 0.5)

            w = c.get('declared_width', 0) or c.get('width', 0)
            h = c.get('declared_height', 0) or c.get('height', 0)
            phi_dim = (
                self._clamp_normalize(w * h, MIN_PIXELS, MAX_PIXELS)
                if w > 0 and h > 0 else 0.5
            )
            phi_aspect = (
                self._gaussian_aspect(w / h)
                if w > 0 and h > 0 else 0.5
            )
            url_keywords = self._extract_keywords_from_url(c['url'])
            phi_rel = self._jaccard(topic_keywords, url_keywords)
            c['_phi_relevance'] = phi_rel  # Stocker pour CLIP re-score

            c['_meta_score'] = (
                pre_weights['source']     * phi_source
                + pre_weights['dimensions'] * phi_dim
                + pre_weights['aspect']     * phi_aspect
                + pre_weights['relevance']  * phi_rel
            )

        candidates.sort(key=lambda x: x.get('_meta_score', 0), reverse=True)
        return candidates

    # =================================================================
    # STAGE 4 : Download parallèle + validation perceptuelle
    # =================================================================
    def _download_and_validate_parallel(
        self, candidates: List[Dict], topic: str
    ) -> List[Dict]:
        results: List[Dict] = []

        with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as pool:
            futures = {}
            for c in candidates:
                url = c['url']
                domain = urlparse(url).netloc
                if not self._circuit.is_healthy(domain):
                    continue
                future = pool.submit(self._download_validate_optimize, c, topic)
                futures[future] = c

            for future in as_completed(futures, timeout=DOWNLOAD_TIMEOUT + 15):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except ImageValidationError as e:
                    c = futures[future]
                    logger.debug(
                        "  ⛔ Rejeté [%s] : %s",
                        c['url'][-40:], str(e)[:60],
                    )
                except Exception as e:
                    c = futures[future]
                    logger.debug(
                        "  ❌ Erreur [%s] : %s",
                        c['url'][-40:], str(e)[:60],
                    )
                    self._circuit.record_failure(urlparse(c['url']).netloc)

        return results

    def _download_validate_optimize(
        self, candidate: Dict, topic: str
    ) -> Optional[Dict]:
        """
        Download, validation perceptuelle (6 critères), sauvegarde.
        Stocke _entropy et _sharpness pour le re-scoring CLIP.
        """
        img_url = candidate['url']
        source = candidate['source']
        session = self._get_session()

        resp = session.get(img_url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()

        ct = resp.headers.get('Content-Type', '').lower()
        if ct and not ct.startswith('image/'):
            raise ImageValidationError(f"MIME invalide : {ct}")

        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        chunks: List[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > max_bytes:
                raise ImageValidationError(f"> {MAX_FILE_SIZE_MB}MB")
            chunks.append(chunk)
        raw_bytes = b''.join(chunks)

        try:
            img = Image.open(io.BytesIO(raw_bytes))
            img.load()
        except Exception as e:
            raise ImageValidationError(f"Image corrompue : {e}")

        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            raise ImageValidationError(f"Trop petite : {width}×{height}")

        ratio = width / height
        if not (MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO):
            raise ImageValidationError(f"Ratio invalide : {ratio:.2f}")

        entropy = img.convert('L').entropy()
        if entropy < MIN_ENTROPY:
            raise ImageValidationError(f"Entropie faible : H={entropy:.1f}")

        sharpness = self._compute_sharpness(img)
        if sharpness < MIN_LAPLACIAN_VAR:
            raise ImageValidationError(f"Image floue : σ²L={sharpness:.0f}")

        # Score sans CLIP (sera remplacé en T4 si CLIP dispo)
        phi_rel = candidate.get('_phi_relevance', 0.0)
        weights = WEIGHTS_WITHOUT_CLIP
        pre_score = (
            weights['source']     * SOURCE_SCORES.get(source, 0.5)
            + weights['dimensions'] * self._clamp_normalize(
                width * height, MIN_PIXELS, MAX_PIXELS)
            + weights['aspect']     * self._gaussian_aspect(ratio)
            + weights['entropy']    * self._clamp_normalize(
                entropy, MIN_ENTROPY, MAX_ENTROPY)
            + weights['sharpness']  * self._clamp_normalize(
                sharpness, MIN_LAPLACIAN_VAR, MAX_LAPLACIAN_VAR)
            + weights['relevance']  * phi_rel
        )

        phash = self._compute_dhash(img)

        if width > TARGET_MAX_WIDTH:
            new_h = int(height * TARGET_MAX_WIDTH / width)
            img = img.resize(
                (TARGET_MAX_WIDTH, new_h), Image.Resampling.LANCZOS
            )
            width, height = TARGET_MAX_WIDTH, new_h

        result = self._save_image(img, img_url, source, width, height)
        result['_final_score']    = pre_score
        result['_phash']          = phash
        result['_entropy']        = entropy     # Pour CLIP re-score
        result['_sharpness']      = sharpness   # Pour CLIP re-score
        result['_phi_relevance']  = phi_rel
        result['_clip_score']     = 0.0
        result['_clip_used']      = False

        self._circuit.record_success(urlparse(img_url).netloc)
        logger.debug(
            "  ✓ %s | %dx%d | H=%.1f | σ²L=%.0f | pre_score=%.3f",
            img_url[-35:], width, height, entropy, sharpness, pre_score,
        )
        return result

    # =================================================================
    # Formules mathématiques
    # =================================================================
    @staticmethod
    def _clamp_normalize(value: float, vmin: float, vmax: float) -> float:
        if vmax <= vmin:
            return 0.5
        return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))

    @staticmethod
    def _gaussian_aspect(ratio: float) -> float:
        return math.exp(-((ratio - OPTIMAL_RATIO) / RATIO_SIGMA) ** 2)

    @staticmethod
    def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    # =================================================================
    # Analyse perceptuelle
    # =================================================================
    @staticmethod
    def _compute_sharpness(img: Image.Image) -> float:
        gray = img.convert('L')
        laplacian = gray.filter(
            ImageFilter.Kernel(
                (3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=0
            )
        )
        if HAS_NUMPY:
            arr = np.array(laplacian, dtype=np.float64)
            return float(np.var(arr))
        pixels = list(laplacian.getdata())
        n = len(pixels)
        if n == 0:
            return 0.0
        mean = sum(pixels) / n
        return sum((p - mean) ** 2 for p in pixels) / n

    @staticmethod
    def _compute_dhash(img: Image.Image, size: int = 9) -> int:
        resized = img.convert('L').resize(
            (size, size - 1), Image.Resampling.LANCZOS
        )
        pixels = list(resized.getdata())
        hash_val = 0
        for row in range(size - 1):
            for col in range(size - 1):
                offset = row * size + col
                hash_val <<= 1
                if pixels[offset] > pixels[offset + 1]:
                    hash_val |= 1
        return hash_val

    @staticmethod
    def _hamming_distance(h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count('1')

    # =================================================================
    # pHash déduplication
    # =================================================================
    def _phash_dedup(self, results: List[Dict]) -> List[Dict]:
        if len(results) <= 1:
            return results
        results.sort(
            key=lambda r: r.get('_final_score', 0), reverse=True
        )
        kept: List[Dict] = []
        for r in results:
            h = r.get('_phash', 0)
            is_dup = any(
                self._hamming_distance(h, k.get('_phash', 0)) < PHASH_THRESHOLD
                for k in kept
            )
            if not is_dup:
                kept.append(r)

        if len(kept) < len(results):
            logger.debug(
                "  pHash dédup : %d → %d candidats",
                len(results), len(kept),
            )
        return kept

    # =================================================================
    # SearXNG Fallback
    # =================================================================
    def _searxng_fallback(
        self, topic: str, category: str
    ) -> Optional[Dict]:
        queries = self._build_searxng_queries(topic, category)
        all_candidates: List[Dict] = []

        with ThreadPoolExecutor(max_workers=MAX_SEARXNG_WORKERS) as pool:
            futures = [
                pool.submit(self._search_searxng_images, q)
                for q in queries
            ]
            for future in as_completed(futures, timeout=SEARXNG_TIMEOUT + 5):
                try:
                    results = future.result()
                    all_candidates.extend(results)
                except Exception as e:
                    logger.debug("  ⚠️ SearXNG échoué : %s", str(e)[:50])

        all_candidates = self._deduplicate_urls(all_candidates)
        if not all_candidates:
            return None
        return self._pipeline(all_candidates, topic)

    def _build_searxng_queries(
        self, topic: str, category: str
    ) -> List[str]:
        terms = self._extract_important_terms(topic)
        primary = ' '.join(terms[:3])
        context = CATEGORY_CONTEXT.get(
            category.upper(), CATEGORY_CONTEXT['GENERAL']
        )
        return [
            f"{primary} {context}",
            f"{primary} HD illustration",
            topic,
        ]

    def _search_searxng_images(self, query: str) -> List[Dict]:
        session = self._get_session()
        try:
            resp = session.get(
                f"{self.searxng_url}/search",
                params={
                    "q": query,
                    "categories": "images",
                    "format": "json",
                    "language": "en-US",
                    "safesearch": "1",
                },
                timeout=SEARXNG_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json().get("results", [])
        except Exception as e:
            logger.error("❌ SearXNG [%s] : %s", query[:50], e)
            return []

        candidates: List[Dict] = []
        for r in raw[:15]:
            img_url = r.get("img_src") or r.get("url")
            if not img_url:
                continue
            if any(pat in img_url.lower() for pat in URL_REJECT_PATTERNS):
                continue
            candidates.append({
                "url": img_url,
                "source": "searxng",
                "source_page": r.get("url", ""),
                "declared_width": r.get("img_width", 0) or 0,
                "declared_height": r.get("img_height", 0) or 0,
                "content_type": "",
                "content_length": 0,
            })

        candidates.sort(
            key=lambda x: x['declared_width'] * x['declared_height'],
            reverse=True,
        )
        return candidates

    # =================================================================
    # Adaptive top-N
    # =================================================================
    @staticmethod
    def _adaptive_top_n(scored: List[Dict], offset: int = 0) -> int:
        if offset >= len(scored):
            return 0
        top_score = scored[offset].get('_meta_score', 0)
        if top_score >= 0.85:
            return 2
        elif top_score >= 0.60:
            return 3
        return TOP_N_DOWNLOAD

    # =================================================================
    # Utilitaires
    # =================================================================
    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        return set(words)

    @staticmethod
    def _extract_keywords_from_url(url: str) -> Set[str]:
        parsed = urlparse(url)
        path = parsed.path.lower()
        for sep in ('/', '_', '-', '.'):
            path = path.replace(sep, ' ')
        return {w for w in path.split() if len(w) > 3 and not w.isdigit()}

    @staticmethod
    def _extract_important_terms(
        topic: str, max_terms: int = 5
    ) -> List[str]:
        words = [w for w in topic.split() if len(w) > 2]
        if not words:
            return []
        max_len = max(len(w) for w in words)
        scored = [
            (w, (len(w) / max_len) * (1 / (1 + 0.3 * i)))
            for i, w in enumerate(words)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scored[:max_terms]]

    @staticmethod
    def _resolve_url(img_url: str, page_url: str) -> str:
        if img_url.startswith("//"):
            return "https:" + img_url
        if img_url.startswith("/"):
            parsed = urlparse(page_url)
            return f"{parsed.scheme}://{parsed.netloc}{img_url}"
        return img_url

    @staticmethod
    def _meta_int(soup: BeautifulSoup, property_name: str) -> int:
        tag = soup.find("meta", property=property_name)
        if tag and tag.get("content"):
            try:
                return int(tag["content"])
            except (ValueError, TypeError):
                return 0
        return 0

    @staticmethod
    def _deduplicate_urls(candidates: List[Dict]) -> List[Dict]:
        seen: Set[str] = set()
        deduped = []
        for c in candidates:
            url = c.get('url')
            if url and url not in seen:
                seen.add(url)
                deduped.append(c)
        return deduped

    def _save_image(
        self,
        img: Image.Image,
        img_url: str,
        source: str,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        url_hash = hashlib.md5(img_url.encode()).hexdigest()[:10]
        filename = f"img_{url_hash}.jpg"
        filepath = os.path.join(self.OUTPUT_DIR, filename)
        img.save(filepath, "JPEG", quality=JPEG_QUALITY,
                 optimize=True, progressive=True)
        file_size_kb = os.path.getsize(filepath) // 1024
        return {
            "local_path":   filepath,
            "original_url": img_url,
            "width":        width,
            "height":       height,
            "file_size_kb": file_size_kb,
            "filename":     filename,
            "source":       source,
        }

    @staticmethod
    def _clean_result(result: Dict) -> Dict:
        return {
            "local_path":   result.get("local_path"),
            "original_url": result.get("original_url"),
            "width":        result.get("width"),
            "height":       result.get("height"),
            "file_size_kb": result.get("file_size_kb"),
            "filename":     result.get("filename"),
            "source":       result.get("source"),
            "score":        result.get("_final_score", 0.0),
            "clip_score":   result.get("_clip_score", 0.0),    # ← NOUVEAU
            "clip_used":    result.get("_clip_used", False),    # ← NOUVEAU
        }
