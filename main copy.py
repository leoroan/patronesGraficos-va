
import json

import cv2

from src.capture import CaptureManager
from src.vision import VisionEngine


def load_config() -> dict:
    """Carga la configuración desde config.json."""

    with open("config.json", "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Punto de entrada principal del programa."""

    # --------------------------------------------------
    # 1. Configuración
    # --------------------------------------------------

    config = load_config()

    capture_config = config["window"]
    hsv_config = config["vision"]["hsv"]
    rectangle_config = config["vision"]["rectangle"]

    min_area = rectangle_config["min_area"]
    epsilon = rectangle_config["epsilon"]

    # --------------------------------------------------
    # 2. Componentes principales
    # --------------------------------------------------

    capture = CaptureManager(
        capture_config
    )

    vision = VisionEngine()

    # --------------------------------------------------
    # 3. Preparación de ventanas OpenCV
    #
    # WINDOW_NORMAL permite redimensionarlas.
    # Se define explícitamente el tamaño y posición de
    # las ventanas auxiliares para evitar estados
    # anteriores de Windows/OpenCV.
    # --------------------------------------------------

    cv2.destroyAllWindows()

    cv2.namedWindow("Captura", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Contornos", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Candidatos", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Mask", 800, 500)
    cv2.resizeWindow("Contornos", 800, 500)
    cv2.resizeWindow("Candidatos", 800, 500)

    cv2.moveWindow("Mask", 50, 50)
    cv2.moveWindow("Contornos", 900, 50)
    cv2.moveWindow("Candidatos", 50, 600)

    # --------------------------------------------------
    # 4. Bucle principal
    # --------------------------------------------------

    while True:

        # --------------------------------------------------
        # 4.1 Captura
        # --------------------------------------------------

        frame = capture.capture_frame()

        # --------------------------------------------------
        # 4.2 Procesamiento HSV
        # --------------------------------------------------

        hsv_frame = vision.to_hsv(
            frame
        )

        mask = vision.create_mask(
            hsv_frame,
            hsv_config["ranges"],
        )

        # --------------------------------------------------
        # 4.3 Detección de contornos
        # --------------------------------------------------

        contours = vision.find_contours(
            mask
        )

        filtered_contours = (
            vision.filter_contours_by_area(
                contours,
                min_area,
            )
        )
        
        median_width = vision.get_median_width(
            filtered_contours
        )
        
        width_tolerance = rectangle_config[
            "width_tolerance"
        ]
        
        width_candidates = (
            vision.filter_contours_by_width(
                filtered_contours,
                median_width,
                width_tolerance,
            )
        ) 
        
        width_candidate_frame = (
            vision.draw_quadrilaterals(
                frame,
                width_candidates,
            )
        )

        # --------------------------------------------------
        # 4.4 Detección de candidatos geométricos
        #
        # Un candidato actualmente debe cumplir:
        #
        # - área mínima
        # - 4 vértices aproximados
        # - convexidad
        # - ángulos aproximadamente rectos
        # --------------------------------------------------

        quadrilaterals = []

        for contour in filtered_contours:

            if vision.is_quadrilateral(
                contour,
                epsilon,
            ):
                quadrilaterals.append(
                    contour
                )

        # --------------------------------------------------
        # 4.5 Generación de imágenes de visualización
        # --------------------------------------------------

        contour_frame = vision.draw_contours(
            frame,
            filtered_contours,
        )
        
        width_candidates = (
            vision.filter_contours_by_width(
                filtered_contours,
                median_width,
                width_tolerance,
            )
        )

        width_candidate_frame = (
            vision.draw_quadrilaterals(
                frame,
                width_candidates,
            )
        )
        
        candidate_display = cv2.resize(
            width_candidate_frame,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA,
        )

        # --------------------------------------------------
        # 4.6 Visualización
        #
        # La captura se reduce solamente para visualizarla.
        # El frame original conserva sus dimensiones.
        # --------------------------------------------------

        display_frame = cv2.resize(
            frame,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA,
        )

        cv2.imshow(
            "Captura",
            display_frame,
        )

        cv2.imshow(
            "Mask",
            mask,
        )

        cv2.imshow(
            "Contornos",
            contour_frame,
        )

        cv2.imshow(
            "Candidatos",
            candidate_display,
        )

        # --------------------------------------------------
        # 4.7 Entrada de teclado
        #
        # S -> muestra información del frame actual.
        # Q -> finaliza el programa.
        # --------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):

            print(
                f"\nContornos: "
                f"{len(contours)} → "
                f"{len(filtered_contours)} "
                f"después del filtro"
            )

            print(
                f"Cuadriláteros candidatos: "
                f"{len(quadrilaterals)}"
            )

            for index, contour in enumerate(
                quadrilaterals,
                start=1,
            ):

                area = cv2.contourArea(
                    contour
                )

                x, y, width, height = (
                    vision.get_bounding_box(
                        contour
                    )
                )

                aspect_ratio = (
                    vision.get_aspect_ratio(
                        contour
                    )
                )

                print(
                    f"  #{index}: "
                    f"área={area:.0f}, "
                    f"x={x}, y={y}, "
                    f"ancho={width}, "
                    f"alto={height}, "
                    f"ratio={aspect_ratio:.2f}"
                )

            # Estadísticas de proporción
            ratios = [
                vision.get_aspect_ratio(
                    contour
                )
                for contour in quadrilaterals
            ]

            if ratios:

                print(
                    f"Ratio mínimo: "
                    f"{min(ratios):.2f}"
                )

                print(
                    f"Ratio máximo: "
                    f"{max(ratios):.2f}"
                )

                print(
                    f"Ratio promedio: "
                    f"{sum(ratios) / len(ratios):.2f}"
                )
                
            median_width = vision.get_median_width(
                filtered_contours
            )
            
            print(
                f"Ancho mediano: {median_width:.2f}px"
            )
            
            print(
                f"Ancho mediano: {median_width:.2f}px"
            )

            print(
                f"Candidatos por ancho: "
                f"{len(width_candidates)}"
            )

        if key == ord("q"):
            break

    # --------------------------------------------------
    # 5. Limpieza
    # --------------------------------------------------

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# Orquesta el programa.

# Debe ser lo más simple posible.

# leer configuración

# ↓

# crear CaptureManager

# ↓

# while

# ↓

# capturar frame

# ↓

# mostrar frame

# sobre HSV
# Cada píxel tiene ahora tres valores:

# pixel
#  ├── H
#  ├── S
#  └── V
# ¿Qué significa cada uno?

# H — Hue

# Representa el tono/color. OpenCV utiliza un rango de 0–179 para Hue.

# S — Saturation

# Representa qué tan intenso o puro es el color. Va de 0–255.

# V — Value

# Representa la luminosidad/intensidad. También 0–255.
