# Minimal Streamlit analytics GUI scaffold
import streamlit as st


st.set_page_config(page_title="Gnosis Analytics Workbench", layout="wide")

# --- Sidebar Hamburger Menu ---
st.sidebar.title("☰ Menu")
page = st.sidebar.radio("Navigation", ["New Experiment", "Analyst"], index=0)

st.title("gnosisPWB - Prompt Workbench")



if page == "New Experiment":
	st.header("Start a New Experiment")
	st.markdown("""
	Configure and launch a new research experiment. Upload a config file or fill out the form below to set up the INFERENCE STEP.
	""")
	import yaml, json
	col1, col2 = st.columns(2)
	with col1:
		st.subheader("Upload Experiment Config")
		uploaded = st.file_uploader("Upload YAML/JSON config", type=["yaml", "yml", "json"])
		config_data = None
		inference_defaults = {"input_csv": "data/prompts.csv", "output_csv": "artifacts/inference_raw.csv", "default_model": "google:default"}
		if uploaded:
			config_text = uploaded.read().decode()
			st.code(config_text, language="yaml" if uploaded.name.endswith("yaml") or uploaded.name.endswith("yml") else "json")
			try:
				if uploaded.name.endswith("json"):
					config_data = json.loads(config_text)
				else:
					config_data = yaml.safe_load(config_text)
				# Try to extract inference step config
				if config_data and "steps" in config_data:
					for step in config_data["steps"]:
						if step.get("component") == "inference":
							inf_cfg = step.get("config", {})
							inference_defaults["input_csv"] = inf_cfg.get("input_csv", inference_defaults["input_csv"])
							inference_defaults["output_csv"] = inf_cfg.get("output_csv", inference_defaults["output_csv"])
							inference_defaults["default_model"] = inf_cfg.get("default_model", inference_defaults["default_model"])
			except Exception as e:
				st.warning(f"Could not parse config: {e}")
	with col2:
		st.subheader("Fill out inference step config:")
		with st.form("inference_form"):
			input_csv = st.text_input("Input CSV", value=inference_defaults["input_csv"])
			output_csv = st.text_input("Output CSV", value=inference_defaults["output_csv"])
			default_model = st.text_input("Default Model", value=inference_defaults["default_model"])
			submitted = st.form_submit_button("Run Inference Step")
			if submitted:
				st.success(f"Would run inference with: input_csv={input_csv}, output_csv={output_csv}, default_model={default_model}")

elif page == "Analyst":
	st.header("Analytics Data Explorer")
	st.markdown("""
	Explore, filter, visualize, and export your experimental data.
	""")



import pandas as pd

import os
from src.analytics.db import FileAnalyticsDataAccess
from src.analytics.db_postgres import PostgresAnalyticsDataAccess

st.header("Analytics Data Explorer")
st.info("Data connection not yet implemented. This is a placeholder using sample data.")


# --- Data Access Layer ---
st.sidebar.header("Data Backend Selection")
backend = st.sidebar.selectbox("Select backend", ["CSV", "PostgreSQL"])

df = None
dal = None
if backend == "CSV":
	DATA_PATH = st.sidebar.text_input(
		"Data file path (CSV)",
		value=os.environ.get("GNOSIS_ANALYTICS_DATA", "analytics_sample.csv")
	)
	dal = FileAnalyticsDataAccess(DATA_PATH, filetype="csv")
	try:
		df = dal.load_data()
	except Exception:
		df = pd.DataFrame({
			"Metric": ["Accuracy", "Precision", "Recall", "F1", "AUC"],
			"Value": [0.92, 0.89, 0.91, 0.90, 0.93],
			"Experiment": ["A", "A", "A", "B", "B"],
			"Date": ["2025-08-20", "2025-08-20", "2025-08-20", "2025-08-21", "2025-08-21"]
		})
elif backend == "PostgreSQL":
	st.sidebar.markdown("**PostgreSQL Connection**")
	pg_user = st.sidebar.text_input("User", value=os.environ.get("PGUSER", "postgres"))
	pg_password = st.sidebar.text_input("Password", type="password", value=os.environ.get("PGPASSWORD", ""))
	pg_host = st.sidebar.text_input("Host", value=os.environ.get("PGHOST", "localhost"))
	pg_port = st.sidebar.text_input("Port", value=os.environ.get("PGPORT", "5432"))
	# Connect to default DB to list databases
	if st.sidebar.button("Connect to Postgres Server"):
		import psycopg2
		try:
			conn = psycopg2.connect(
				dbname="postgres",
				user=pg_user,
				password=pg_password,
				host=pg_host,
				port=pg_port
			)
			with conn.cursor() as cur:
				cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
				dbs = [row[0] for row in cur.fetchall()]
			conn.close()
			st.session_state["pg_dbs"] = dbs
			st.sidebar.success("Connected to server. Select a database.")
		except Exception as e:
			st.sidebar.error(f"Postgres connection failed: {e}")
	dbs = st.session_state.get("pg_dbs", [])
	pg_db = st.sidebar.selectbox("Database", dbs) if dbs else st.sidebar.text_input("Database", value=os.environ.get("PGDATABASE", "analytics"))
	# List tables in selected DB
	tables = []
	if dbs and pg_db:
		import psycopg2
		try:
			conn = psycopg2.connect(
				dbname=pg_db,
				user=pg_user,
				password=pg_password,
				host=pg_host,
				port=pg_port
			)
			with conn.cursor() as cur:
				cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
				tables = [row[0] for row in cur.fetchall()]
			conn.close()
		except Exception as e:
			st.sidebar.error(f"Could not list tables: {e}")
	pg_table = st.sidebar.selectbox("Table", tables) if tables else st.sidebar.text_input("Table", value="metrics")
	# Connect and load data
	if st.sidebar.button("Load Table Data"):
		try:
			dal = PostgresAnalyticsDataAccess(
				user=pg_user,
				password=pg_password,
				host=pg_host,
				port=pg_port,
				dbname=pg_db,
				table=pg_table
			)
			df = dal.load_data()
			st.sidebar.success(f"Loaded data from {pg_db}.{pg_table}.")
		except Exception as e:
			st.sidebar.error(f"Failed to load table: {e}")
	if df is None:
		df = pd.DataFrame({
			"Metric": ["Accuracy", "Precision", "Recall", "F1", "AUC"],
			"Value": [0.92, 0.89, 0.91, 0.90, 0.93],
			"Experiment": ["A", "A", "A", "B", "B"],
			"Date": ["2025-08-20", "2025-08-20", "2025-08-20", "2025-08-21", "2025-08-21"]
		})

# --- Filtering UI ---

# --- Data Filtering, Visualization, and Export ---
if df is not None and not df.empty:
	exp_options = df["Experiment"].unique().tolist()
	selected_exp = st.multiselect("Select Experiment(s)", exp_options, default=exp_options)
	metric_options = df["Metric"].unique().tolist()
	selected_metrics = st.multiselect("Select Metric(s)", metric_options, default=metric_options)
	date_options = df["Date"].unique().tolist()
	selected_dates = st.multiselect("Select Date(s)", date_options, default=date_options)

	# Filter DataFrame
	filtered_df = df[
		df["Experiment"].isin(selected_exp) &
		df["Metric"].isin(selected_metrics) &
		df["Date"].isin(selected_dates)
	]

	st.subheader("Filtered Data Table")
	st.dataframe(filtered_df, use_container_width=True)

	# --- Visualization ---
	st.subheader("Metric Visualization")
	chart_type = st.selectbox("Chart Type", ["Bar", "Line"])
	if not filtered_df.empty:
		chart_df = filtered_df.pivot(index="Metric", columns="Experiment", values="Value")
		if chart_type == "Bar":
			st.bar_chart(chart_df)
		else:
			st.line_chart(chart_df)
	else:
		st.warning("No data to display for selected filters.")

	# --- Export ---
	st.subheader("Export Data")
	csv = filtered_df.to_csv(index=False).encode('utf-8')
	st.download_button(
		label="Download filtered data as CSV",
		data=csv,
		file_name="filtered_analytics_data.csv",
		mime="text/csv"
	)
else:
	st.warning("No data loaded. Please connect to a backend and load data.")
