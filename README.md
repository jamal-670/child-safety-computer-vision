# SafeGuard

SafeGuard processes video input to detect possible falls using YOLO pose
estimation. By default, it processes all supported videos in `data/raw/`.

## Run locally

Install the Python dependencies and put one or more `.mp4`, `.avi`, `.mov`, or
`.mkv` videos in `data/raw/`. Then run from the project directory:

```sh
python -m pip install -r requirements.txt
python -m src.main
```

To process one video instead of the whole folder:

```sh
python -m src.main --source data/raw/your-video.mp4
```

The application opens a display window while processing. Press `q` to stop.

## Run with Docker

Install Docker with Docker Compose on the machine that will run SafeGuard. From
the project directory, build the image:

```sh
docker compose build
```

The image includes the Python dependencies and the `yolov8n-pose.pt` model
weights. The weights are downloaded during the image build, so building
requires an internet connection.

Put video files in `data/raw/` on the host. Process all supported videos in
that folder:

```sh
docker compose run --rm safeguard
```

Or process one video:

```sh
docker compose run --rm safeguard \
  --headless --source /app/data/raw/your-video.mp4
```

The container runs headlessly and prints telemetry and fall alerts to the
terminal. The Compose service mounts the host's `data/` directory read-only at
`/app/data`.

On Linux, to process the default webcam instead, run:

```sh
docker run --rm --device=/dev/video0:/dev/video0 safeguard:latest
```

The existing `requirements.txt` is used by the Docker build to install the
dependencies; you do not need to install them separately on the other machine.
