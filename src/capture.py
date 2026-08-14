# CaptureManager

# Única responsabilidad:

# Obtener imágenes.

# No analiza.

# No detecta.

# No sabe qué es una vela.

# No sabe qué es un color.

# Solo entrega un frame.

import mss
import pygetwindow as gw
import numpy as np

class CaptureManager:
    """Gestiona la localización y captura de una ventana."""

    def __init__(self, window_config: dict) -> None:
        self.window_title = window_config["title"]
        self.roi = window_config.get("roi")

        self.sct = mss.MSS()

    def find_window(self) -> gw.Win32Window:
        """Busca la ventana configurada."""

        windows = gw.getWindowsWithTitle(self.window_title)

        if not windows:
            raise RuntimeError(
                f"No se encontró ninguna ventana con el título "
                f"'{self.window_title}'"
            )

        return windows[0]

    def get_window_region(self) -> dict[str, int]:
        """Obtiene la posición y dimensiones actuales de la ventana."""

        window = self.find_window()

        if window.width <= 0 or window.height <= 0:
            raise RuntimeError(
                f"La ventana '{self.window_title}' no tiene "
                "dimensiones válidas."
            )

        return {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height,
        }

    def get_capture_regions(self) -> list[dict[str, int]]:
        """Calcula las regiones de captura a partir de las ROI configuradas."""

        window_region = self.get_window_region()

        if self.roi is None:
            return [window_region]

        regions = []

        for roi in self.roi:
            regions.append(
                {
                    "left": window_region["left"] + roi["x"],
                    "top": window_region["top"] + roi["y"],
                    "width": roi["width"],
                    "height": roi["height"],
                }
            )

        return regions

    def capture_frame(self) -> np.ndarray:
        """Captura la primera región configurada."""
    
        regions = self.get_capture_regions()
    
        screenshot = self.sct.grab(regions[0])
    
        return np.array(screenshot)
    
    def capture_full_window(self) -> np.ndarray:
        """Captura la ventana completa, ignorando temporalmente la ROI."""

        region = self.get_window_region()

        screenshot = self.sct.grab(
            region
        )

        return np.array(
            screenshot
        )
        
    def set_roi(
        self,
        roi: dict[str, int],
    ) -> None:
        """Actualiza la ROI utilizada para la captura."""

        self.roi = [roi]