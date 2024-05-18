import cv2
import os
import json
import numpy as np
from pydantic import BaseModel, Field
from typing import Literal
from image_processing import optical_flow_diff_sequence, frame_diff, image_subtraction


class Parameters(BaseModel):
    overlay_alpha: float = Field(
        0.3, description="Alpha blending value for overlay", ge=0, le=1
    )
    optical_flow_weight: float = Field(
        0.5, description="Weight for optical flow results", ge=0, le=1
    )
    frame_diff_weight: float = Field(
        0.5, description="Weight for frame difference results", ge=0, le=1
    )
    image_subtraction_weight: float = Field(
        0.5, description="Weight for image subtraction results", ge=0, le=1
    )
    pyr_scale: float = Field(0.5, description="Pyramid scale factor", ge=0, le=1)
    levels: int = Field(1, description="Number of pyramid levels", ge=1, le=5)
    winsize: int = Field(5, description="Window size for optical flow", ge=1, le=30)
    iterations: int = Field(
        1, description="Number of iterations for optical flow", ge=1, le=10
    )
    poly_n: int = Field(
        5, description="Size of pixel neighborhood for polynomial expansion", ge=1, le=7
    )
    poly_sigma: float = Field(
        0.5,
        description="Standard deviation of the Gaussian used for polynomial expansion",
        ge=0,
        le=2,
    )
    num_images: int = Field(
        2, description="Number of images to compare (between 2 and 4)", ge=2, le=4
    )
    blend_mode: Literal["normal", "multiply", "burn"] = Field(
        "normal", description="Blending mode for overlay"
    )

    def to_trackbar_values(self):
        values = {}
        for field_name, field in self.__fields__.items():
            if field_name == "blend_mode":
                continue
            value = getattr(self, field_name)
            if isinstance(value, float):
                value = int(value * 10)
            values[field_name] = value
        return values

    def from_trackbar_values(self, values):
        for field_name, field in self.__fields__.items():
            if field_name == "blend_mode":
                continue
            value = values[field_name]
            if isinstance(getattr(self, field_name), float):
                value = value / 10
            setattr(self, field_name, value)


class ImageProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = sorted(
            [
                f
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            ]
        )
        self.idx = 0
        self.images = []
        self.current_image_file = ""
        self.params = Parameters()

        self.create_windows()
        self.load_parameters()
        self.update_images()

    def create_windows(self):
        cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Control", cv2.WINDOW_NORMAL)

        self.create_trackbars()
        self.create_buttons()
        self.create_blend_mode_dropdown()

    def create_trackbars(self):
        for field_name, field in self.params.__fields__.items():
            if field_name == "blend_mode":
                continue
            initial = self.params.to_trackbar_values()[field_name]
            if isinstance(getattr(self.params, field_name), float):
                maxval = int(field.field_info.le * 10)
            else:
                maxval = field.field_info.le
            cv2.createTrackbar(field_name, "Control", initial, maxval, lambda x: None)

    def create_buttons(self):
        cv2.createButton("Update", self.update_image, None, cv2.QT_PUSH_BUTTON, 1)
        cv2.createButton(
            "Save Params",
            lambda *args: self.save_parameters(),
            None,
            cv2.QT_PUSH_BUTTON,
            1,
        )
        cv2.createButton(
            "Load Params",
            lambda *args: self.load_parameters(),
            None,
            cv2.QT_PUSH_BUTTON,
            1,
        )
        cv2.createButton(
            "Next", lambda *args: self.next_image(), None, cv2.QT_PUSH_BUTTON, 1
        )
        cv2.createButton(
            "Previous", lambda *args: self.prev_image(), None, cv2.QT_PUSH_BUTTON, 1
        )

    def create_blend_mode_dropdown(self):
        blend_modes = ["normal", "multiply", "burn"]
        self.blend_mode_index = blend_modes.index(self.params.blend_mode)
        self.blend_mode_max_index = len(blend_modes) - 1

        cv2.createTrackbar(
            "Blend Mode",
            "Control",
            self.blend_mode_index,
            self.blend_mode_max_index,
            self.update_blend_mode,
        )

    def update_blend_mode(self, index):
        blend_modes = ["normal", "multiply", "burn"]
        self.params.blend_mode = blend_modes[index]

    def resize_images(self):
        if not self.images:
            return
        first_shape = self.images[0].shape
        for i in range(1, len(self.images)):
            if self.images[i].shape != first_shape:
                self.images[i] = cv2.resize(
                    self.images[i], (first_shape[1], first_shape[0])
                )

    def blend_images(self, base_img, overlay_img, mode):
        if mode == "multiply":
            return cv2.multiply(base_img, overlay_img)
        elif mode == "burn":
            return cv2.subtract(base_img, overlay_img)
        else:
            return cv2.addWeighted(
                base_img,
                1 - self.params.overlay_alpha,
                overlay_img,
                self.params.overlay_alpha,
                0,
            )

    def update_image(self, *args):
        if len(self.images) < 2:
            return

        trackbar_values = {
            field_name: cv2.getTrackbarPos(field_name, "Control")
            for field_name in self.params.__fields__
            if field_name != "blend_mode"
        }
        self.params.from_trackbar_values(trackbar_values)

        self.resize_images()

        combined_result = np.zeros_like(self.images[0])

        if self.params.optical_flow_weight > 0:
            optical_flow_params = {
                k: getattr(self.params, k)
                for k in [
                    "pyr_scale",
                    "levels",
                    "winsize",
                    "iterations",
                    "poly_n",
                    "poly_sigma",
                ]
            }
            optical_flow_result = optical_flow_diff_sequence(
                self.images, **optical_flow_params
            )
            if optical_flow_result.shape != combined_result.shape:
                optical_flow_result = cv2.resize(
                    optical_flow_result,
                    (combined_result.shape[1], combined_result.shape[0]),
                )
            combined_result = cv2.addWeighted(
                combined_result,
                1,
                optical_flow_result,
                self.params.optical_flow_weight,
                0,
            )

        if self.params.frame_diff_weight > 0:
            frame_diff_result = frame_diff(self.images[0], self.images[1])
            if frame_diff_result.shape != combined_result.shape:
                frame_diff_result = cv2.resize(
                    frame_diff_result,
                    (combined_result.shape[1], combined_result.shape[0]),
                )
            combined_result = cv2.addWeighted(
                combined_result, 1, frame_diff_result, self.params.frame_diff_weight, 0
            )

        if self.params.image_subtraction_weight > 0:
            image_subtraction_result = image_subtraction(self.images[0], self.images[1])
            if image_subtraction_result.shape != combined_result.shape:
                image_subtraction_result = cv2.resize(
                    image_subtraction_result,
                    (combined_result.shape[1], combined_result.shape[0]),
                )
            combined_result = cv2.addWeighted(
                combined_result,
                1,
                image_subtraction_result,
                self.params.image_subtraction_weight,
                0,
            )

        base_img = self.images[-1]
        overlay = self.blend_images(base_img, combined_result, self.params.blend_mode)

        # Draw the image filename on the overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            overlay,
            self.current_image_file,
            (10, overlay.shape[0] - 10),
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Overlay", overlay)

    def save_parameters(self):
        with open("parameters.json", "w") as f:
            json.dump(self.params.dict(), f)
        print("Parameters saved to parameters.json")

    def load_parameters(self):
        try:
            with open("parameters.json", "r") as f:
                params_dict = json.load(f)
                self.params = Parameters(**params_dict)

            for field_name in self.params.__fields__:
                if field_name == "blend_mode":
                    continue
                initial = self.params.to_trackbar_values()[field_name]
                cv2.setTrackbarPos(field_name, "Control", initial)

            blend_modes = ["normal", "multiply", "burn"]
            self.update_blend_mode(blend_modes.index(self.params.blend_mode))

            print("Parameters loaded from parameters.json")
        except FileNotFoundError:
            print("parameters.json not found. Using default parameters.")

    def update_images(self):
        num_images = int(self.params.num_images)
        if self.idx < len(self.image_files) - num_images:
            self.images = [
                cv2.imread(os.path.join(self.folder_path, self.image_files[i]))
                for i in range(self.idx, self.idx + num_images)
                if i < len(self.image_files)
            ]
            self.current_image_file = self.image_files[
                self.idx
            ]  # Update the current image filename
            self.update_image()

    def next_image(self):
        if self.idx < len(self.image_files) - self.params.num_images:
            self.idx += 1
            self.update_images()

    def prev_image(self):
        if self.idx > 0:
            self.idx -= 1
            self.update_images()

    def run(self):
        cv2.resizeWindow(
            "Control", 500, 600
        )  # Resize the control window to ensure it's visible

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # Escape key
                break
            elif key == 81:  # Left arrow key
                self.prev_image()
            elif key == 83:  # Right arrow key
                self.next_image()

        cv2.destroyAllWindows()


def main(folder_path: str):
    processor = ImageProcessor(folder_path)
    processor.run()


if __name__ == "__main__":
    import typer

    typer.run(main)
