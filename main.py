import json

import cv2

from src.capture import CaptureManager
from src.vision import VisionEngine
from src.roi_selector import ROISelector

def load_config() -> dict:
    """Carga la configuración desde config.json."""

    with open("config.json", "r", encoding="utf-8") as file:
        return json.load(file)

def save_roi(
    config: dict,
    roi: dict[str, int],
) -> None:
    """Guarda la ROI seleccionada en config.json."""

    config["window"]["roi"] = [roi]

    with open(
        "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False,
        )


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
    width_tolerance = rectangle_config["width_tolerance"]

    # --------------------------------------------------
    # 2. Componentes principales
    # --------------------------------------------------

    capture = CaptureManager(
        capture_config
    )

    vision = VisionEngine()
    
    roi_selector = ROISelector()

    # --------------------------------------------------
    # 3. Ventanas OpenCV
    #
    # Se crean explícitamente para evitar problemas con
    # posiciones/estados residuales de ejecuciones previas.
    # --------------------------------------------------

    cv2.destroyAllWindows()

    cv2.namedWindow(
        "Captura",
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        "Mask",
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        "Contornos",
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        "Candidatos",
        cv2.WINDOW_NORMAL,
    )

    # Tamaños cómodos para las ventanas auxiliares.
    cv2.resizeWindow(
        "Mask",
        800,
        500,
    )

    cv2.resizeWindow(
        "Contornos",
        800,
        500,
    )

    cv2.resizeWindow(
        "Candidatos",
        800,
        500,
    )

    # Posiciones de las ventanas.
    cv2.moveWindow(
        "Mask",
        50,
        50,
    )

    cv2.moveWindow(
        "Contornos",
        900,
        50,
    )

    cv2.moveWindow(
        "Candidatos",
        50,
        600,
    )

    # --------------------------------------------------
    # 4. Bucle principal
    # --------------------------------------------------

    while True:

        # --------------------------------------------------
        # 4.1 Captura
        #
        # CaptureManager obtiene la ROI configurada
        # respecto de la ventana objetivo.
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

        # --------------------------------------------------
        # 4.4 Filtrado básico por área
        #
        # Eliminamos regiones demasiado pequeñas para
        # considerarlas de interés.
        # --------------------------------------------------

        filtered_contours = (
            vision.filter_contours_by_area(
                contours,
                min_area,
            )
        )

        # --------------------------------------------------
        # 4.5 Determinar ancho típico
        #
        # El ancho de referencia se obtiene dinámicamente
        # de los propios contornos encontrados.
        # No usamos un valor fijo en píxeles.
        # --------------------------------------------------

        median_width = vision.get_median_width(
            filtered_contours
        )

        # --------------------------------------------------
        # 4.6 Filtrar por ancho relativo
        #
        # Los candidatos deben tener un ancho cercano al
        # ancho típico encontrado en esta ROI.
        # --------------------------------------------------

        width_candidates = (
            vision.filter_contours_by_width(
                filtered_contours,
                median_width,
                width_tolerance,
            )
        )

        # --------------------------------------------------
        # 4.7 Visualización de todos los contornos
        # --------------------------------------------------

        contour_frame = vision.draw_contours(
            frame,
            filtered_contours,
        )

        # --------------------------------------------------
        # 4.8 Visualización de candidatos
        #
        # Actualmente "Candidatos" representa las regiones
        # que pasan el filtro de ancho relativo.
        # --------------------------------------------------

        candidate_frame = vision.draw_quadrilaterals(
            frame,
            width_candidates,
        )

        # --------------------------------------------------
        # 4.9 Preparar imágenes para visualización
        #
        # La reducción es SOLO visual.
        #
        # El frame original mantiene la resolución completa
        # de la ROI para el procesamiento.
        # --------------------------------------------------

        display_frame = cv2.resize(
            frame,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA,
        )

        candidate_display = cv2.resize(
            candidate_frame,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA,
        )

        # --------------------------------------------------
        # 4.10 Mostrar resultados
        # --------------------------------------------------

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
        # 4.11 Teclado
        #
        # S → muestra estadísticas del frame actual.
        # Q → finaliza el programa.
        # --------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            full_frame = capture.capture_full_window()

            selected_roi = roi_selector.select(
                full_frame
            )

            if selected_roi is not None:
                capture.set_roi(
                    selected_roi
                )

                save_roi(
                    config,
                    selected_roi,
                )

                print(
                    "ROI actualizada:",
                    selected_roi,
                )

            continue

        if key == ord("s"):

            print(
                f"\nContornos: "
                f"{len(contours)} → "
                f"{len(filtered_contours)} "
                f"después del filtro"
            )

            print(
                f"Ancho mediano: "
                f"{median_width:.2f}px"
            )

            print(
                f"Candidatos por ancho: "
                f"{len(width_candidates)}"
            )

            for index, contour in enumerate(
                width_candidates,
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
                
            x_gap = vision.get_median_x_gap(
                width_candidates
            )

            print(
                f"Separación X mediana: "
                f"{x_gap:.2f}px"
            )    
            
            x_gaps = vision.get_x_gaps(
                width_candidates
            )
            
            print(
                f"Separaciones X: {x_gaps}"
            )

        if key == ord("q"):
            break

    # --------------------------------------------------
    # 5. Limpieza
    # --------------------------------------------------

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()