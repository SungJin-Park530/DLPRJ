from pathlib import Path
from typing import Any

import streamlit as st


@st.cache_data
def get_model_configs() -> dict[str, str]:
	"""Return the configured paths for the application's model slots."""
	return {
		"model_1": "",
		"model_2": "",
	}


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str) -> Any:
	"""Load and retain one trained model instance for the Streamlit process."""
	from ultralytics import YOLO

	return YOLO(model_path)


@st.cache_data
def get_model_status(model_path: str) -> dict[str, str | bool]:
	"""Return display-safe metadata for a configured model path."""
	path = Path(model_path)
	return {
		"path": str(path),
		"configured": bool(model_path),
		"available": path.is_file(),
	}


def main() -> None:
	"""Render the base application layout."""
	st.set_page_config(
		page_title="Streamlit App",
		page_icon="",
		layout="wide",
		initial_sidebar_state="expanded",
	)

	with st.sidebar:
		st.title("Menu")
		st.divider()
		st.caption("Navigation will be added here.")

	st.title("Streamlit App")
	st.caption("A starting point for your deployed application.")
	st.divider()

	content_area, details_area = st.columns([2, 1], gap="large")
	with content_area:
		st.subheader("Main content")
		st.info("Add the primary app experience here.")

	with details_area:
		st.subheader("Model status")
		for model_name, model_path in get_model_configs().items():
			status = get_model_status(model_path)
			label = model_name.replace("_", " ").title()
			if not status["configured"]:
				st.caption(f"{label}: path not configured")
			elif status["available"]:
				st.success(f"{label}: ready")
			else:
				st.warning(f"{label}: file not found")


if __name__ == "__main__":
	main()
