# Deploying This API on Render

## Project Files

Keep these files in your GitHub repository:

```text
Server.py
requirements.txt
render.yaml
best.pt
```

The `best.pt` file contains your trained YOLO model data. The server loads it through the `MODEL_PATH` environment variable, which defaults to `best.pt`.

## Render Settings

If you deploy manually from the Render dashboard, use:

```text
Runtime: Python
Build Command: pip install -r requirements.txt
Start Command: uvicorn Server:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```text
PYTHON_VERSION=3.12.10
MODEL_PATH=best.pt
```

After deployment, open:

```text
https://your-service-name.onrender.com/docs
```

## Notes

- Use `opencv-python-headless` on Render because cloud servers do not need OpenCV desktop GUI features.
- If `best.pt` is too large for GitHub, use Git LFS or store the model in external storage and download it during deployment.
- Render's filesystem is temporary between deploys unless you attach a persistent disk.
