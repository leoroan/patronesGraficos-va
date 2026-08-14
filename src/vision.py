# Única responsabilidad:

# Transformar imágenes.

# Por ejemplo

# Frame
# ↓

# HSV

# ↓

# Máscara

# ↓

# Contornos

# ↓

# Objetos encontrados

# Nunca decide.

# Nunca cuenta.

# Nunca interpreta.


import cv2
import numpy as np


class VisionEngine:
    """Procesa imágenes capturadas."""

    def to_hsv(self, frame: np.ndarray) -> np.ndarray:
        """Convierte una imagen BGRA a HSV."""

        bgr_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR,
        )

        return cv2.cvtColor(
            bgr_frame,
            cv2.COLOR_BGR2HSV,
        )
        
    def split_hsv(
        self,
        hsv_frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Separa una imagen HSV en sus canales H, S y V."""

        return cv2.split(hsv_frame)

    def create_mask(
        self,
        hsv_frame: np.ndarray,
        ranges: list[dict],
    ) -> np.ndarray:
        """Crea una máscara combinando múltiples rangos HSV."""

        mask = np.zeros(
            hsv_frame.shape[:2],
            dtype=np.uint8,
        )

        for color_range in ranges:
            lower_bound = np.array(
                color_range["lower"],
                dtype=np.uint8,
            )

            upper_bound = np.array(
                color_range["upper"],
                dtype=np.uint8,
            )

            range_mask = cv2.inRange(
                hsv_frame,
                lower_bound,
                upper_bound,
            )

            mask = cv2.bitwise_or(
                mask,
                range_mask,
            )

        return mask
        
    def mask_statistics(self, mask: np.ndarray) -> dict[str, float]:
      """Calcula estadísticas básicas de una máscara binaria."""

      total_pixels = mask.size
      selected_pixels = cv2.countNonZero(mask)

      percentage = (selected_pixels / total_pixels) * 100

      return {
          "total_pixels": total_pixels,
          "selected_pixels": selected_pixels,
          "percentage": percentage,
      }
      
    def erode(
        self,
        mask: np.ndarray,
        kernel_size: int = 3,
        iterations: int = 1,
    ) -> np.ndarray:
        """Reduce las regiones blancas de una máscara."""

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        return cv2.erode(
            mask,
            kernel,
            iterations=iterations,
        )

    def dilate(
        self,
        mask: np.ndarray,
        kernel_size: int = 3,
        iterations: int = 1,
    ) -> np.ndarray:
        """Expande las regiones blancas de una máscara."""

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        return cv2.dilate(
            mask,
            kernel,
            iterations=iterations,
        )
        
    def hsv_statistics(self, hsv_frame: np.ndarray) -> dict[str, float]:
        """Obtiene valores mínimos y máximos de cada canal HSV."""
    
        hue = hsv_frame[:, :, 0]
        saturation = hsv_frame[:, :, 1]
        value = hsv_frame[:, :, 2]
    
        return {
            "h_min": float(np.min(hue)),
            "h_max": float(np.max(hue)),
            "s_min": float(np.min(saturation)),
            "s_max": float(np.max(saturation)),
            "v_min": float(np.min(value)),
            "v_max": float(np.max(value)),
        }

    def morphology_open(
        self,
        mask: np.ndarray,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """Aplica una operación Opening sobre una máscara."""

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        return cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

    def morphology_close(
        self,
        mask: np.ndarray,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """Aplica una operación Closing sobre una máscara."""

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )
          
    def find_contours(
        self,
        mask: np.ndarray,
    ) -> list[np.ndarray]:
        """Detecta los contornos presentes en una máscara."""
    
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        return contours
    
    def filter_contours_by_area(
        self,
        contours: list[np.ndarray],
        min_area: float,
    ) -> list[np.ndarray]:
        """Conserva únicamente los contornos con área suficiente."""

        return [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= min_area
        ]
        
    def draw_contours(
        self,
        frame: np.ndarray,
        contours: list[np.ndarray],
    ) -> np.ndarray:
        """Dibuja los contornos sobre una copia del frame."""

        output = frame.copy()

        cv2.drawContours(
            output,
            contours,
            -1,
            (0, 255, 0, 255),
            2,
        )

        return output

    def approximate_contour(
        self,
        contour: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        """Aproxima un contorno a un polígono."""

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        return cv2.approxPolyDP(
            contour,
            epsilon * perimeter,
            True,
        )
        
    def create_test_rectangle(
        self,
        width: int = 800,
        height: int = 500,
    ) -> np.ndarray:
        """Crea una máscara artificial con un rectángulo blanco."""
    
        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )
    
        cv2.rectangle(
            mask,
            (200, 100),
            (600, 400),
            255,
            -1,
        )
    
        return mask
    
    def is_quadrilateral(
        self,
        contour: np.ndarray,
        epsilon: float,
        angle_tolerance: float = 15.0,
    ) -> bool:
        """Comprueba si un contorno se aproxima a un rectángulo."""
    
        approximated = self.approximate_contour(
            contour,
            epsilon,
        )
    
        if len(approximated) != 4:
            return False
    
        if not cv2.isContourConvex(approximated):
            return False
    
        return self.has_right_angles(
            approximated,
            angle_tolerance,
        )
    
    def get_bounding_box(
        self,
        contour: np.ndarray,
    ) -> tuple[int, int, int, int]:
        """Obtiene el rectángulo delimitador de un contorno."""
    
        return cv2.boundingRect(contour)
    
    def get_aspect_ratio(
        self,
        contour: np.ndarray,
    ) -> float:
        """Calcula la relación ancho/alto del bounding box."""

        _, _, width, height = self.get_bounding_box(contour)

        if height == 0:
            return 0.0

        return width / height
    
    def draw_quadrilaterals(
        self,
        frame: np.ndarray,
        contours: list[np.ndarray],
    ) -> np.ndarray:
        """Dibuja candidatos cuadriláteros y sus bounding boxes."""
    
        output = frame.copy()
    
        for contour in contours:
            cv2.drawContours(
                output,
                [contour],
                -1,
                (0, 255, 0, 255),
                2,
            )
    
            x, y, width, height = cv2.boundingRect(
                contour
            )
    
            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                (255, 0, 0, 255),
                1,
            )
    
        return output
    
    def is_convex(
        self,
        contour: np.ndarray,
    ) -> bool:
        """Comprueba si un contorno es convexo."""

        return cv2.isContourConvex(contour)
    
    def calculate_angle(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
        point_c: np.ndarray,
    ) -> float:
        """Calcula el ángulo ABC en grados."""

        vector_ba = point_a - point_b
        vector_bc = point_c - point_b

        norm_ba = np.linalg.norm(vector_ba)
        norm_bc = np.linalg.norm(vector_bc)

        if norm_ba == 0 or norm_bc == 0:
            return 0.0

        cosine = np.dot(vector_ba, vector_bc) / (
            norm_ba * norm_bc
        )

        cosine = np.clip(cosine, -1.0, 1.0)

        return float(
            np.degrees(
                np.arccos(cosine)
            )
        )
        
    def has_right_angles(
        self,
        polygon: np.ndarray,
        tolerance: float = 15.0,
    ) -> bool:
        """Comprueba si un cuadrilátero tiene ángulos aproximadamente rectos."""

        points = polygon.reshape(4, 2)

        angles = []

        for index in range(4):
            point_a = points[index - 1]
            point_b = points[index]
            point_c = points[(index + 1) % 4]

            angle = self.calculate_angle(
                point_a,
                point_b,
                point_c,
            )

            angles.append(angle)

        return all(
            abs(angle - 90.0) <= tolerance
            for angle in angles
        )
        
    def get_median_width(
            self,
            contours: list[np.ndarray],
        ) -> float:
            """Obtiene el ancho mediano de los bounding boxes."""

            if not contours:
                return 0.0

            widths = [
                self.get_bounding_box(contour)[2]
                for contour in contours
            ]

            return float(np.median(widths))
        
    def filter_contours_by_width(
        self,
        contours: list[np.ndarray],
        reference_width: float,
        tolerance: float,
    ) -> list[np.ndarray]:
        """Filtra contornos según una tolerancia relativa al ancho de referencia."""

        if reference_width <= 0:
            return []

        minimum = reference_width * (1 - tolerance)
        maximum = reference_width * (1 + tolerance)

        return [
            contour
            for contour in contours
            if minimum
            <= self.get_bounding_box(contour)[2]
            <= maximum
        ]
        
    def get_x_positions(
        self,
        contours: list[np.ndarray],
    ) -> list[int]:
        """Obtiene las posiciones X de los contornos ordenadas."""

        positions = [
            self.get_bounding_box(contour)[0]
            for contour in contours
        ]

        return sorted(positions)

    def get_x_gaps(
        self,
        contours: list[np.ndarray],
    ) -> list[int]:
        """Calcula las distancias horizontales entre candidatos vecinos."""

        positions = self.get_x_positions(contours)

        if len(positions) < 2:
            return []

        return [
            positions[index + 1] - positions[index]
            for index in range(len(positions) - 1)
        ]
        
    def get_median_x_gap(
        self,
        contours: list[np.ndarray],
    ) -> float:
        """Obtiene la separación horizontal mediana."""
    
        gaps = self.get_x_gaps(contours)
    
        if not gaps:
            return 0.0
    
        return float(np.median(gaps))