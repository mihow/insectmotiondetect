# cli_analyze_images.py
import typer
import cv2
import os
from rich.progress import Progress
from image_processing import optical_flow_diff, frame_diff, image_subtraction


def process_images(
    folder_path: str,
    output_path: str,
    window_size: int,
    overlay_alpha: float,
    optical_flow_weight: float,
    frame_diff_weight: float,
    image_subtraction_weight: float,
    use_last_frame: bool,
    pyr_scale: float,
    levels: int,
    winsize: int,
    iterations: int,
    poly_n: int,
    poly_sigma: float,
):
    image_files = sorted(
        [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]
    )

    with Progress() as progress:
        task = progress.add_task(
            "[green]Processing images...", total=len(image_files) - window_size + 1
        )

        for i in range(len(image_files) - window_size + 1):
            img1 = cv2.imread(os.path.join(folder_path, image_files[i]))
            img2 = cv2.imread(os.path.join(folder_path, image_files[i + 1]))
            img3 = (
                cv2.imread(os.path.join(folder_path, image_files[i + 2]))
                if window_size > 2
                else None
            )

            optical_flow_result = optical_flow_diff(
                img1,
                img2,
                img3,
                pyr_scale,
                levels,
                winsize,
                iterations,
                poly_n,
                poly_sigma,
            )
            frame_diff_result = frame_diff(img1, img2)
            image_subtraction_result = image_subtraction(img1, img2)

            combined_result = cv2.addWeighted(
                optical_flow_result,
                optical_flow_weight,
                frame_diff_result,
                frame_diff_weight,
                0,
            )
            combined_result = cv2.addWeighted(
                combined_result,
                1,
                image_subtraction_result,
                image_subtraction_weight,
                0,
            )

            base_img = img3 if use_last_frame and img3 is not None else img2
            overlay = cv2.addWeighted(
                base_img, 1 - overlay_alpha, combined_result, overlay_alpha, 0
            )
            cv2.imwrite(os.path.join(output_path, f"motion_overlay_{i}.png"), overlay)

            progress.advance(task)


def main(
    folder_path: str,
    output_path: str,
    window_size: int = 3,
    overlay_alpha: float = 0.3,
    optical_flow_weight: float = 0.5,
    frame_diff_weight: float = 0.5,
    image_subtraction_weight: float = 0.5,
    use_last_frame: bool = True,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
):
    process_images(
        folder_path,
        output_path,
        window_size,
        overlay_alpha,
        optical_flow_weight,
        frame_diff_weight,
        image_subtraction_weight,
        use_last_frame,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
    )


if __name__ == "__main__":
    typer.run(main)
