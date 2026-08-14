# mostrar imagen
#       ↓
# recibir mouse
#       ↓
# dibujar rectángulo
#       ↓
# devolver ROI(s)

import cv2
import numpy as np


class ROISelector:
    """Permite seleccionar una región rectangular mediante el mouse."""

    def __init__(self, window_name: str = "Seleccionar ROI") -> None:
        self.window_name = window_name

        self.start_point: tuple[int, int] | None = None
        self.end_point: tuple[int, int] | None = None

        self.dragging = False
        self.confirmed = False

        self.scale = 1.0

    def _mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        """Gestiona la selección mediante el mouse."""

        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (
                int(x / self.scale),
                int(y / self.scale),
            )

            self.end_point = self.start_point
            self.dragging = True

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_point = (
                int(x / self.scale),
                int(y / self.scale),
            )

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.end_point = (
                int(x / self.scale),
                int(y / self.scale),
            )

            self.dragging = False

    def select(
        self,
        frame: np.ndarray,
    ) -> dict[str, int] | None:
        """
        Permite seleccionar una ROI.

        ENTER confirma.
        ESC cancela.
        """

        height, width = frame.shape[:2]

        # Reducimos solamente la imagen de selección para
        # que una ventana demasiado grande siga siendo manejable.
        max_width = 1200
        max_height = 800

        scale_x = max_width / width
        scale_y = max_height / height

        self.scale = min(
            1.0,
            scale_x,
            scale_y,
        )

        display = cv2.resize(
            frame,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_AREA,
        )

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )

        cv2.setMouseCallback(
            self.window_name,
            self._mouse_callback,
        )

        while True:
            preview = display.copy()

            if (
                self.start_point is not None
                and self.end_point is not None
            ):
                x1, y1 = self.start_point
                x2, y2 = self.end_point

                left = min(x1, x2)
                top = min(y1, y2)
                right = max(x1, x2)
                bottom = max(y1, y2)

                cv2.rectangle(
                    preview,
                    (
                        int(left * self.scale),
                        int(top * self.scale),
                    ),
                    (
                        int(right * self.scale),
                        int(bottom * self.scale),
                    ),
                    (0, 255, 0, 255),
                    2,
                )

            cv2.imshow(
                self.window_name,
                preview,
            )

            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):  # ENTER
                if (
                    self.start_point is None
                    or self.end_point is None
                ):
                    continue

                x1, y1 = self.start_point
                x2, y2 = self.end_point

                left = min(x1, x2)
                top = min(y1, y2)
                right = max(x1, x2)
                bottom = max(y1, y2)

                roi_width = right - left
                roi_height = bottom - top

                if roi_width <= 0 or roi_height <= 0:
                    continue

                self.confirmed = True

                cv2.destroyWindow(
                    self.window_name
                )

                return {
                    "x": left,
                    "y": top,
                    "width": roi_width,
                    "height": roi_height,
                }

            if key == 27:  # ESC
                cv2.destroyWindow(
                    self.window_name
                )

                return None