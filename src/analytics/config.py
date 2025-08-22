def validate_config(cfg):
    required_keys = ["input_csv", "output_csv", "group_col"]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
    # Optionally, add type checks or more validation here