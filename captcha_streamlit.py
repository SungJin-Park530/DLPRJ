from pathlib import Path
from typing import Any

import gdown
import streamlit as st

MODEL_DIRECTORY = Path(__file__).parent / "weights"

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
		prepare_model_files(progress_bar)
		progress_bar.progress(100, text="Model files ready.")
		progress_bar.empty()
	else:
		prepare_model_files()

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
				st.subheader("분석 이미지")
				st.caption("이미지가 이 영역에 표시됩니다.")

		with prediction_column:
			with st.container(height=460, border=True):
				st.subheader("예측 결과")
				st.write("클래스명 (nn.n%)")
				st.write("클래스명 (nn.n%)")
				st.write("클래스명 (nn.n%)")

		random_column, upload_column, _ = st.columns([1, 1, 3], gap="small")
		with random_column:
			st.button("랜덤 이미지 호출", disabled=True)

		with upload_column:
			uploaded_file = st.file_uploader(
				"이미지 업로드",
				type=["jpg", "jpeg", "png", "webp"],
				label_visibility="collapsed",
			)
			if uploaded_file is not None:
				if uploaded_file.type.startswith("image/"):
					st.success(f"{uploaded_file.name} 파일이 선택되었습니다.")
				else:
					st.error("이미지 파일만 업로드할 수 있습니다.")

	with results_tab:
		st.subheader("학습 결과 및 비교")
		st.info("모델 학습 결과와 비교 시각자료가 이 영역에 표시됩니다.")


if __name__ == "__main__":
	main()
