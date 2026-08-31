from io import BytesIO
from pathlib import Path
from random import choice
from time import perf_counter
from typing import Any

import gdown
import streamlit as st
from PIL import Image

MODEL_DIRECTORY = Path(__file__).parent / "weights"
SAMPLE_DIRECTORY = Path(__file__).parent / "samples"

@st.cache_data
def get_model_configs() -> dict[str, dict[str, str]]:
	"""Return the configured paths for the application's model slots."""
	return {
		"medium": {
			"filename": "v8m_best.pt",
			"file_id": "1AEBqkRXDIe5tidL22Y06yJ6bEeQ_RExj",
		},
		"nano": {
			"filename": "v8n_best.pt",
			"file_id": "1Ec8dk-7auVrsHiousXMzESpWgiFlzVTX",
		},
	}


def get_missing_model_names() -> list[str]:
	"""Return models whose local weight files are not available yet."""
	return [
		model_name
		for model_name, config in get_model_configs().items()
		if not (MODEL_DIRECTORY / config["filename"]).is_file()
	]


def prepare_model_files(progress_bar: Any | None = None) -> dict[str, str]:
	"""Download missing model files and return their local paths."""
	MODEL_DIRECTORY.mkdir(exist_ok=True)
	model_paths: dict[str, str] = {}
	model_configs = get_model_configs()
	completed_models = 0
	total_models = len(model_configs)

	for model_name, config in model_configs.items():
		model_path = MODEL_DIRECTORY / config["filename"]
		if not model_path.is_file():
			if progress_bar is not None:
				progress_bar.progress(
					int(completed_models / total_models * 100),
					text=f"Downloading {model_name.title()} model...",
				)

			def update_progress(downloaded_bytes: int, total_bytes: int | None) -> None:
				if progress_bar is None or total_bytes is None or total_bytes == 0:
					return
				model_progress = min(downloaded_bytes / total_bytes, 1.0)
				progress_bar.progress(
					int((completed_models + model_progress) / total_models * 100),
					text=f"Downloading {model_name.title()} model...",
				)

			downloaded_path = gdown.download(
				id=config["file_id"],
				output=str(model_path),
				quiet=False,
				progress=update_progress,
			)
			if downloaded_path is None or not model_path.is_file():
				raise RuntimeError(f"Could not download {model_name} model.")
		model_paths[model_name] = str(model_path)
		completed_models += 1

	return model_paths


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str) -> Any:
	"""Load and retain one trained model instance for the Streamlit process."""
	from ultralytics import YOLO

	return YOLO(model_path)


@st.cache_data
def get_sample_image_paths() -> tuple[str, ...]:
	"""Return image files available for random sample selection."""
	image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
	return tuple(
		str(image_path)
		for image_path in SAMPLE_DIRECTORY.iterdir()
		if image_path.suffix.lower() in image_extensions
	)


def clear_prediction() -> None:
	"""Clear results when the selected source image changes."""
	st.session_state.pop("prediction_image", None)
	st.session_state.pop("prediction_rows", None)
	st.session_state.pop("no_detections", None)
	st.session_state.pop("inference_time_ms", None)


def select_random_image() -> None:
	"""Select one image from the local sample directory."""
	sample_paths = get_sample_image_paths()
	if not sample_paths:
		st.session_state["image_error"] = "samples 폴더에 사용할 이미지가 없습니다."
		return
	st.session_state["selected_image"] = choice(sample_paths)
	st.session_state["selected_image_name"] = Path(st.session_state["selected_image"]).name
	st.session_state.pop("image_error", None)
	clear_prediction()


def select_uploaded_image() -> None:
	"""Store a validated uploaded image in session state."""
	uploaded_file = st.session_state.get("image_uploader")
	if uploaded_file is None:
		return
	if not uploaded_file.type.startswith("image/"):
		st.session_state["image_error"] = "이미지 파일만 업로드할 수 있습니다."
		return
	st.session_state["selected_image"] = uploaded_file.getvalue()
	st.session_state["selected_image_name"] = uploaded_file.name
	st.session_state.pop("image_error", None)
	clear_prediction()


def get_selected_image() -> Image.Image | None:
	"""Open the selected path or uploaded bytes as an RGB image."""
	image_source = st.session_state.get("selected_image")
	if image_source is None:
		return None
	if isinstance(image_source, bytes):
		return Image.open(BytesIO(image_source)).convert("RGB")
	return Image.open(image_source).convert("RGB")


def run_prediction(model_paths: dict[str, str]) -> None:
	"""Run inference with the selected cached YOLO model."""
	selected_image = get_selected_image()
	if selected_image is None:
		st.session_state["prediction_error"] = "먼저 분석할 이미지를 선택하세요."
		return

	model_name = st.session_state["model_selector"]
	model = load_model(model_paths[model_name])
	start_time = perf_counter()
	result = model(selected_image, verbose=False)[0]
	st.session_state["inference_time_ms"] = (perf_counter() - start_time) * 1_000
	st.session_state["prediction_image"] = result.plot()[:, :, ::-1]
	st.session_state["no_detections"] = len(result.boxes) == 0
	st.session_state["prediction_rows"] = sorted(
		[
			f"{result.names[int(class_id)]} ({confidence * 100:.1f}%)"
			for class_id, confidence in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
		],
		key=lambda row: float(row.rsplit("(", 1)[1][:-2]),
		reverse=True,
	)
	st.session_state.pop("prediction_error", None)


def main() -> None:
	"""Render the base application layout."""
	st.set_page_config(
		page_title="주행 환경 이미지 분석 서비스",
		page_icon="",
		layout="wide",
		initial_sidebar_state="expanded",
	)
	missing_models = get_missing_model_names()
	if missing_models:
		st.title("Preparing application")
		progress_bar = st.progress(0, text="Preparing model files...")
		model_paths = prepare_model_files(progress_bar)
		progress_bar.progress(100, text="Model files ready.")
		progress_bar.empty()
	else:
		model_paths = prepare_model_files()

	with st.sidebar:
		st.title("Menu")
		st.divider()
		st.caption("Navigation will be added here.")

	st.title("주행 환경 이미지 분석 서비스")
	st.divider()

	main_tab, results_tab = st.tabs(["이미지 분석", "학습 결과 및 비교"])
	with main_tab:
		st.subheader("표지판 객체 탐지")
		image_column, prediction_column = st.columns([3, 2], gap="large")

		with image_column:
			with st.container(height=460, border=True):
				if "prediction_image" in st.session_state:
					st.image(st.session_state["prediction_image"], use_container_width=True)
				elif selected_image := get_selected_image():
					st.image(selected_image, use_container_width=True)
				else:
					st.subheader("분석 이미지")
					st.caption("이미지를 선택하면 이 영역에 표시됩니다.")

		with prediction_column:
			with st.container(height=460, border=True):
				st.subheader("예측 결과")
				if st.session_state.get("no_detections"):
					st.error("예측된 표지판 객체가 없습니다.")
				elif "prediction_rows" in st.session_state:
					for prediction_row in st.session_state["prediction_rows"]:
						st.success(prediction_row)
				else:
					st.caption("예측 결과가 이 영역에 표시됩니다.")

		random_column, upload_column, model_column, predict_column = st.columns([1, 2, 1, 1], gap="small")
		with random_column:
			st.button("랜덤 이미지 호출", on_click=select_random_image)

		with upload_column:
			st.file_uploader(
				"이미지 업로드",
				type=["jpg", "jpeg", "png", "webp"],
				label_visibility="collapsed",
				key="image_uploader",
				on_change=select_uploaded_image,
			)

		with model_column:
			st.selectbox(
				"모델 선택",
				options=["nano", "medium"],
				format_func=lambda model_name: model_name.title(),
				label_visibility="collapsed",
				key="model_selector",
			)

		with predict_column:
			st.button("예측하기", on_click=run_prediction, args=(model_paths,))

		if inference_time_ms := st.session_state.get("inference_time_ms"):
			st.caption(f"예측 시간: {inference_time_ms:.2f} ms")

		if image_error := st.session_state.get("image_error"):
			st.error(image_error)
		elif prediction_error := st.session_state.get("prediction_error"):
			st.warning(prediction_error)

	with results_tab:
		st.subheader("학습 결과 및 비교")
		st.info("모델 학습 결과와 비교 시각자료가 이 영역에 표시됩니다.")


if __name__ == "__main__":
	main()
