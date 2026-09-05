import base64
from io import BytesIO
from pathlib import Path
from random import choice
from time import perf_counter
from typing import Any
from llm_handler import generate_safety_guide
from audio_handler import text_to_speech_bytes
from streamlit_mic_recorder import mic_recorder
from audio_handler import speech_to_text_from_bytes

import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

MODEL_DIRECTORY = Path(__file__).parent / "weights"
SAMPLE_DIRECTORY = Path(__file__).parent / "samples"
CONTENT_MAX_WIDTH_PX = 1000
IMAGE_PANEL_HEIGHT_PX = 460
MODEL_EVALUATION_METRICS = {
	"medium": {"Precision": 0.9474, "Recall": 0.9122, "mAP50": 0.9750},
	"nano": {"Precision": 0.9252, "Recall": 0.8714, "mAP50": 0.9430},
}

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
	st.session_state.pop("safety_guide", None)
	st.session_state.pop("detected_objects", None)


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


def render_analysis_image(image: Image.Image | Any) -> None:
	"""Render an image centered and contained within the analysis panel."""
	if not isinstance(image, Image.Image):
		image = Image.fromarray(image)
	image_buffer = BytesIO()
	image.convert("RGB").save(image_buffer, format="PNG")
	encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
	st.markdown(
		f"""
		<div class="analysis-image-frame">
			<img src="data:image/png;base64,{encoded_image}" alt="분석 이미지">
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_model_metric_comparison() -> None:
	"""Render the vertical grouped bar chart from the model evaluation."""
	metric_names = list(next(iter(MODEL_EVALUATION_METRICS.values())).keys())
	model_names = list(MODEL_EVALUATION_METRICS.keys())
	x_positions = np.arange(len(metric_names))
	bar_width = 0.35
	figure, axis = plt.subplots(figsize=(10, 5))

	for index, model_name in enumerate(model_names):
		metric_values = [MODEL_EVALUATION_METRICS[model_name][metric_name] for metric_name in metric_names]
		offset = (index - 0.5) * bar_width
		bars = axis.bar(x_positions + offset, metric_values, bar_width, label=model_name.title())
		axis.bar_label(bars, fmt="%.3f", padding=3)

	axis.set_xlabel("Metric")
	axis.set_ylabel("Score")
	axis.set_xticks(x_positions, metric_names)
	axis.set_ylim(0.7, 1.0)
	axis.grid(axis="y", linestyle="--", alpha=0.5)
	axis.legend()
	figure.tight_layout()
	st.pyplot(figure)
	plt.close(figure)


def render_sidebar() -> None:
	"""Render LLM controls independently from the prediction action."""
	st.sidebar.title("LLM 파라미터 조정")
	st.sidebar.slider(
		"Temperature",
		min_value=0.0,
		max_value=1.0,
		value=0.7,
		step=0.1,
		key="temperature",
	)
	st.sidebar.selectbox(
		"프롬프트 버전 선택",
		[
			"기본 안전 가이드 (safety_guide_prompt.md)",
			"간결 모드 (short_prompt.md)",
		],
		index=0,
		key="prompt_option",
	)


def run_prediction(model_paths: dict[str, str]) -> None:
	"""Run inference with the selected cached YOLO model."""
	model_name = st.session_state["model_selector"]
	if model_name == "모델 선택...":
		st.session_state["prediction_error"] = "예측에 사용할 모델을 선택하세요."
		return

	selected_image = get_selected_image()
	if selected_image is None:
		st.session_state["prediction_error"] = "먼저 분석할 이미지를 선택하세요."
		return

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

	# 1~3. 탐지된 클래스 이름 및 개수 집계
	detected_objects: dict[str, int] = {}
	for class_id in result.boxes.cls.tolist():
		class_name = result.names[int(class_id)]
		detected_objects[class_name] = detected_objects.get(class_name, 0) + 1
	st.session_state["detected_objects"] = detected_objects

	prompt_option = st.session_state["prompt_option"]
	
	prompt_path_map = {
		"기본 안전 가이드 (safety_guide_prompt.md)": "prompts/safety_guide_prompt.md",
		"간결 모드 (short_prompt.md)": "prompts/short_prompt.md"
	}
	selected_prompt_path = prompt_path_map[prompt_option]

	# 4~6. LLM 안전 가이드 생성
	with st.spinner("AI 안전 가이드를 생성하는 중..."):
		st.session_state["safety_guide"] = generate_safety_guide(
			detected_objects,
			temperature=st.session_state["temperature"],
			prompt_file_path=selected_prompt_path
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
	render_sidebar()
	st.markdown(
		f"""
		<style>
			.stMainBlockContainer {{
				max-width: {CONTENT_MAX_WIDTH_PX}px;
			}}
			.analysis-image-frame {{
				display: flex;
				height: calc({IMAGE_PANEL_HEIGHT_PX}px - 2rem);
				align-items: center;
				justify-content: center;
				overflow: hidden;
			}}
			.analysis-image-frame img {{
				max-width: 100%;
				max-height: 100%;
				width: auto !important;
				height: auto !important;
				object-fit: contain;
			}}
		</style>
		""",
		unsafe_allow_html=True,
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

	st.title("주행 환경 이미지 분석 서비스")
	st.divider()

	main_tab, results_tab = st.tabs(["이미지 분석", "학습 결과 및 비교"])
	with main_tab:
		image_column, prediction_column = st.columns([3, 2], gap="large")

		with image_column:
			with st.container(height=IMAGE_PANEL_HEIGHT_PX, border=True, key="analysis-image"):
				if "prediction_image" in st.session_state:
					render_analysis_image(st.session_state["prediction_image"])
				elif selected_image := get_selected_image():
					render_analysis_image(selected_image)
				else:
					st.subheader("분석 이미지")
					st.caption("이미지를 선택하면 이 영역에 표시됩니다.")

		with prediction_column:
			with st.container(height=460, border=False):
				st.file_uploader(
					"이미지 업로드",
					type=["jpg", "jpeg", "png", "webp"],
					label_visibility="collapsed",
					key="image_uploader",
					on_change=select_uploaded_image,
				)
				st.selectbox(
					"모델 선택",
					options=["모델 선택...", "nano", "medium"],
					format_func=lambda model_name: model_name.title() if model_name != "모델 선택..." else model_name,
					label_visibility="collapsed",
					key="model_selector",
				)
				button_column, random_column = st.columns(2, gap="small")
				with button_column:
					st.button("예측하기", on_click=run_prediction, args=(model_paths,), use_container_width=True)
				with random_column:
					st.button("랜덤 이미지", on_click=select_random_image, use_container_width=True)

				if inference_time_ms := st.session_state.get("inference_time_ms"):
					st.caption(f"예측 시간: {inference_time_ms:.2f} ms")

				if image_error := st.session_state.get("image_error"):
					st.error(image_error)
				if prediction_error := st.session_state.get("prediction_error"):
					st.error(prediction_error)
				elif st.session_state.get("no_detections"):
					st.error("예측된 표지판 객체가 없습니다.")
				elif "prediction_rows" in st.session_state:
					for prediction_row in st.session_state["prediction_rows"]:
						st.success(prediction_row)
				else:
					st.info("예측 결과가 이 영역에 표시됩니다.")

		if safety_guide := st.session_state.get("safety_guide"):
			st.info(safety_guide)

			# 음성 출력 추가
			audio_bytes = text_to_speech_bytes(safety_guide)
			if audio_bytes:
				st.audio(audio_bytes, format="audio/mp3", autoplay=True)
		
		# 음성 입력 추가
		st.markdown("---")
		st.subheader("🎤 음성 명령 / 질문 입력")

		# 브라우저 마이크 녹음 버튼 생성
		audio = mic_recorder(
			start_prompt="🎤 녹음 시작",
			stop_prompt="⏹️ 녹음 중지",
			format="wav",
			key="mic_recorder"
		)

		# 마이크 녹음 데이터가 들어오면 STT 실행
		if audio:
			# 녹음된 WAV 바이너리 데이터 추출
			audio_bytes = audio['bytes']
			
			with st.spinner("음성을 텍스트로 변환하는 중..."):
				stt_result = speech_to_text_from_bytes(audio_bytes)
			
			# 변환된 텍스트 화면 출력
			st.success(f" 인식된 질문: **{stt_result}**")

	with results_tab:
		render_model_metric_comparison()

		metrics_dataframe = pd.DataFrame(MODEL_EVALUATION_METRICS).T
		metrics_dataframe.index = metrics_dataframe.index.str.title()
		st.bar_chart(metrics_dataframe, horizontal=True)

		metric_table = pd.DataFrame(
			{
				"비교 항목": ["모델 파일 크기", "학습 시간", "예측 시간"],
				"Nano": ["6,088 KB", "약 5~6분", "약 30ms"],
				"Medium": ["50,840 KB", "약 10~12분", "약 70ms"],
			}
		)
		with st.container(horizontal_alignment="center"):
			st.dataframe(metric_table, hide_index=True, width=360)


if __name__ == "__main__":
	main()
